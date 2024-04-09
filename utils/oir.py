from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import abc
import utils.basic_utils as basic_utils
import shutil
from torch.optim.adam import Adam
from PIL import Image
from configs.oir_config import GUIDANCE_SCALE, device, NUM_DDIM_STEPS, ldm_stable
import time
from tqdm import trange
import os


@torch.no_grad()
def oir(
    model,
    prompt:  List[str],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    latent: Optional[torch.FloatTensor] = None,
    return_type='image',
    all_latents=None,
    image_mask=None,
    ddim_inversion=None,
    use_reinversion=False,
    reinversion_steps=0,
    prompt_changes=[],
    max_sweet_point=0,
    right_to_left_1_point=0,
    reassembly_step=0,

):

    batch_size = len(prompt)
    height = width = 512
    
    text_input = model.tokenizer(
        prompt,
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    text_embeddings = model.text_encoder(text_input.input_ids.to(model.device))[0]
    max_length = text_input.input_ids.shape[-1]

    latent, latents = basic_utils.init_latent(latent, model, height, width, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    latent_mask = {}
    if image_mask is not None:
        
        latent_mask['right_to_left_1_point'] = image_mask['right_to_left_1_point'].unsqueeze(1).to(latents.device)
        latent_mask['right_to_left_1_point'] = torch.nn.functional.interpolate(
                latent_mask['right_to_left_1_point'].float(), 
                size=(latents.shape[2], latents.shape[3]),
                mode="nearest",
            ) > 0.5
        latent_mask['right_to_left_1_point'] = latent_mask['right_to_left_1_point'].repeat(1, latents.shape[1], 1, 1).int()   

        latent_mask['reassembly_step'] = image_mask['reassembly_step'].unsqueeze(1).to(latents.device)
        latent_mask['reassembly_step'] = torch.nn.functional.interpolate(
                latent_mask['reassembly_step'].float(), 
                size=(latents.shape[2], latents.shape[3]),
                mode="nearest",
            ) > 0.5
        latent_mask['reassembly_step'] = latent_mask['reassembly_step'].repeat(1, latents.shape[1], 1, 1).int()   

        latent_mask[prompt_changes[0]] = image_mask[prompt_changes[0]].unsqueeze(1).to(latents.device)
        latent_mask[prompt_changes[0]] = torch.nn.functional.interpolate(
                latent_mask[prompt_changes[0]].float(), 
                size=(latents.shape[2], latents.shape[3]),
                mode="nearest",
            ) > 0.5
        latent_mask[prompt_changes[0]] = latent_mask[prompt_changes[0]].repeat(1, latents.shape[1], 1, 1).int()  

        latent_mask[prompt_changes[1]] = image_mask[prompt_changes[1]].unsqueeze(1).to(latents.device)
        latent_mask[prompt_changes[1]] = torch.nn.functional.interpolate(
                latent_mask[prompt_changes[1]].float(), 
                size=(latents.shape[2], latents.shape[3]),
                mode="nearest",
            ) > 0.5
        latent_mask[prompt_changes[1]] = latent_mask[prompt_changes[1]].repeat(1, latents.shape[1], 1, 1).int()  

        latent_mask['background'] = image_mask['background'].unsqueeze(1).to(latents.device)
        latent_mask['background'] = torch.nn.functional.interpolate(
                latent_mask['background'].float(), 
                size=(latents.shape[2], latents.shape[3]),
                mode="nearest",
            ) > 0.5
        latent_mask['background'] = latent_mask['background'].repeat(1, latents.shape[1], 1, 1).int()  
        

    for i, t in enumerate(tqdm(model.scheduler.timesteps[-max_sweet_point:])):
        
        num_of_timesteps = len(model.scheduler.timesteps[-max_sweet_point:])
        right_to_left_1_point_area_latent_save_point_denoise_axis = num_of_timesteps - right_to_left_1_point
        reassembly_step_area_latent_save_point_denoise_axis = num_of_timesteps - reassembly_step
        
        context = torch.cat([text_embeddings[0].unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1), text_embeddings])

        latents = basic_utils.diffusion_step(model, latents, context, t, guidance_scale, low_resource=False)


        if i == right_to_left_1_point_area_latent_save_point_denoise_axis:
            latents[1, :, :, :] = all_latents[right_to_left_1_point]

        if i == reassembly_step_area_latent_save_point_denoise_axis:
            latents = replace_2_inversion_unmask_area_to_denoise_unmask_area(
                model,
                i,
                latents,
                all_latents,
                latent_mask,
                reassembly_step,
                reassembly_step_area_latent_save_point_denoise_axis,
                prompt_changes,
            )
            if use_reinversion:
                latents = reinversion_and_denoise(
                    model,
                    latents,
                    context,
                    ddim_inversion,
                    reinversion_steps,
                    reassembly_step,
                    guidance_scale,
                    latent_mask,
                )

    image = basic_utils.latent2image(model.vae, latents) if return_type == 'image' else latents
    return image, latent

def replace_2_inversion_unmask_area_to_denoise_unmask_area(
        model,
        i,
        latents,
        all_latents,
        latent_mask,
        unmask_area_latent_save_point,
        unmask_area_latent_save_point_denoise_axis,
        prompt_changes,

):
    cat_latents = latents[1].unsqueeze(0) * latent_mask[prompt_changes[0]]
    cake_latents = latents[2].unsqueeze(0) * latent_mask[prompt_changes[1]]
    background_latents = all_latents[unmask_area_latent_save_point] * latent_mask['background']
    cat_cake_background_latents = cat_latents + cake_latents + background_latents
    latents[3, :, :, :] = cat_cake_background_latents
    return latents


def reinversion_and_denoise(
        model,
        latents,
        context,
        ddim_inversion,
        reinversion_steps,
        unmask_area_latent_save_point,
        guidance_scale,
        latent_mask,
):
    latents = ddim_inversion.reinversion(latents, unmask_area_latent_save_point, reinversion_steps)
    for t in model.scheduler.timesteps[-(reinversion_steps + unmask_area_latent_save_point): -unmask_area_latent_save_point]:
        latents = basic_utils.diffusion_step(model, latents, context, t, guidance_scale, low_resource=False,)
    return latents
        

    
    # if verbose:
    #     basic_utils.view_images(images, save_path=save_path, file_name=file_name)
    # return images, x_t

