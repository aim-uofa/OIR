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
    generator = None, # generator: Optional[torch.Genwerator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    return_type='image',

    use_negative_prompt_inversion=True,

    all_latents=None,
    mask_area_inversion_point=0,
    unmask_area_latent_save_point=0,

    image_mask=None,
    mask_at_where='latent', # pixel
    unmask_area_recover_method='guided_latent',# guided_latent, inversion_latent
    use_cfg_mask=False,

    ddim_inversion=None,
    use_reinversion=False,
    reinversion_steps=0,
    replace_all_unmask_area_after_replace_point=False,

    prompt_changes=[],
    max_sweet_point=0,
    right_to_left_1_point=0,
    reassembly_step=0,
    blended_all_step_after_reassembly_step=False,

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
    if uncond_embeddings is None:
        uncond_input = model.tokenizer(
            [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
        )
        uncond_embeddings_ = model.text_encoder(uncond_input.input_ids.to(model.device))[0]
    else:
        uncond_embeddings_ = None

    latent, latents = basic_utils.init_latent(latent, model, height, width, generator, batch_size)
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
        
        if use_negative_prompt_inversion:
            context = torch.cat([text_embeddings[0].unsqueeze(0).repeat(text_embeddings.shape[0], 1, 1), text_embeddings])
        else:
            context = torch.cat([uncond_embeddings[i].expand(*text_embeddings.shape), text_embeddings]) if uncond_embeddings_ is None else torch.cat([uncond_embeddings_, text_embeddings])

        latents = basic_utils.diffusion_step(model, latents, context, t, guidance_scale, low_resource=False)


        if i == right_to_left_1_point_area_latent_save_point_denoise_axis:
            latents[1, :, :, :] = all_latents[right_to_left_1_point]

        if i == reassembly_step_area_latent_save_point_denoise_axis or (blended_all_step_after_reassembly_step and i > reassembly_step_area_latent_save_point_denoise_axis):
            latents = replace_2_inversion_unmask_area_to_denoise_unmask_area(
                model,
                i,
                latents,
                all_latents,
                latent_mask,
                mask_at_where,
                reassembly_step,
                reassembly_step_area_latent_save_point_denoise_axis,
                unmask_area_recover_method,
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
                    use_cfg_mask,
                )

    image = basic_utils.latent2image(model.vae, latents) if return_type == 'image' else latents
    return image, latent

def replace_2_inversion_unmask_area_to_denoise_unmask_area(
        model,
        i,
        latents,
        all_latents,
        latent_mask,
        mask_at_where,
        unmask_area_latent_save_point,
        unmask_area_latent_save_point_denoise_axis,
        unmask_area_recover_method,
        prompt_changes,

):
    if mask_at_where == 'pixel':
        denoise_point_to_image = torch.from_numpy(basic_utils.latent2image(model.vae, latents))

        if unmask_area_recover_method == 'guided_latent':
            inversion_point_to_image = torch.from_numpy(basic_utils.latent2image(model.vae, latents[0].unsqueeze(0))) # 用原始prompt的background作为现在的background, 可以改成all_latents[i]
        elif unmask_area_recover_method == 'inversion_latent':
            inversion_point_to_image = torch.from_numpy(basic_utils.latent2image(model.vae, all_latents[unmask_area_latent_save_point])) # 用原始prompt的background作为现在的background, 可以改成all_latents[i]

        image_mask = image_mask.unsqueeze(3).repeat(1, 1, 1, 3)
        new_images = (denoise_point_to_image * image_mask + inversion_point_to_image * (1 - image_mask)).numpy()
        temporal_latents = None
        for i in range(new_images.shape[0]):
            new_image_latent = basic_utils.image2latent(model.vae, new_images[i])
            temporal_latents = new_image_latent if temporal_latents is None else torch.concat([temporal_latents, new_image_latent], dim=0)
        latents = temporal_latents
    elif mask_at_where == 'latent':
        
        if unmask_area_recover_method == 'guided_latent':
            latents = latents * latent_mask + latents[0].unsqueeze(0) * (1 - latent_mask)
        elif unmask_area_recover_method == 'inversion_latent':
            cat_latents = latents[1].unsqueeze(0) * latent_mask[prompt_changes[0]]
            cake_latents = latents[2].unsqueeze(0) * latent_mask[prompt_changes[1]]
            background_latents = all_latents[unmask_area_latent_save_point] * latent_mask['background']
            cat_cake_background_latents = cat_latents + cake_latents + background_latents
            latents[3, :, :, :] = cat_cake_background_latents
    return latents


def reinversion_and_denoise(
        model,
        latents,
        # controller,
        context,
        ddim_inversion,
        reinversion_steps,
        unmask_area_latent_save_point,
        guidance_scale,
        latent_mask,
        use_cfg_mask,
):
    latents = ddim_inversion.reinversion(latents, unmask_area_latent_save_point, reinversion_steps)
    for t in model.scheduler.timesteps[-(reinversion_steps + unmask_area_latent_save_point): -unmask_area_latent_save_point]:
        latents = basic_utils.diffusion_step(model, latents, context, t, guidance_scale, low_resource=False,)
    return latents
        

def dynamic_run_and_display(
    prompts, 
    latent=None, 
    run_baseline=False, 
    generator=None, 
    uncond_embeddings=None, 
    verbose=True,

    save_path='/home/yangzhen/code/DynamicInversion/outputs/debug/',
    file_name='test.png',
    use_negative_prompt_inversion=True,

    method='search metric',
    all_latents=None,
    all_latents_mask=None,

    mask_area_inversion_point=0,
    unmask_area_latent_save_point=0,
    image_mask=None,
    mask_at_where='latent',
    unmask_area_recover_method='guided_latent',
    use_cfg_mask=False,

    ddim_inversion=None,
    use_reinversion=False,
    reinversion_steps=0,
    replace_all_unmask_area_after_replace_point=False,

    max_sweet_point=0,
    right_to_left_1_point=0,
    reassembly_step=0,
    blended_all_step_after_reassembly_step=False, 
    prompt_changes=[]
):
    if method == 'local dynamic inversion':
        images, x_t = oir(
            ldm_stable, 
            prompts, 
            latent=latent, 
            num_inference_steps=NUM_DDIM_STEPS, 
            guidance_scale=GUIDANCE_SCALE,
            generator=generator, 
            uncond_embeddings=uncond_embeddings,
            use_negative_prompt_inversion=use_negative_prompt_inversion,

            all_latents=all_latents,
            image_mask=image_mask,
            mask_at_where=mask_at_where,
            unmask_area_recover_method=unmask_area_recover_method,

            use_cfg_mask=use_cfg_mask,

            ddim_inversion=ddim_inversion,
            use_reinversion=use_reinversion,
            reinversion_steps=reinversion_steps,
            replace_all_unmask_area_after_replace_point=replace_all_unmask_area_after_replace_point,

            max_sweet_point=max_sweet_point,
            right_to_left_1_point=right_to_left_1_point,
            reassembly_step=reassembly_step,
            blended_all_step_after_reassembly_step=blended_all_step_after_reassembly_step,
            prompt_changes=prompt_changes,
        )

    
    if verbose:
        basic_utils.view_images(images, save_path=save_path, file_name=file_name)
    return images, x_t

