from typing import Optional, Union, Tuple, List, Callable, Dict
from tqdm.notebook import tqdm
import torch, abc, shutil, time, os
from diffusers import StableDiffusionPipeline, DDIMScheduler
import torch.nn.functional as nnf
import numpy as np
import utils.basic_utils as basic_utils
from PIL import Image
from configs.basic_config import GUIDANCE_SCALE, device, NUM_DDIM_STEPS, ldm_stable
from tqdm import trange

def change_all_masks_shape(mask, latent):
    mask = mask.unsqueeze(1).to(latent.device)
    mask = torch.nn.functional.interpolate(
        mask.float(),
        size=(latent.shape[2], latent.shape[3]),
        mode='nearest',
    ) > 0.5
    mask = mask.repeat(1, latent.shape[1], 1, 1).int()
    return mask

@torch.no_grad()
def oir(
    model,
    prompts,
    optimal_inversion_steps,
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    latent: Optional[torch.FloatTensor] = None,
    return_type='image',
    all_latents=None,
    all_masks=None,
    ddim_inversion=None,
    reinversion_steps=0,
    prompt_changes=[],
    max_optimal_inversion_step=0,
    right_to_left_1_point=0,
    reassembly_step=0,
    height=512,
    width=512,
):
    batch_size = len(prompts)
    latent, latents = basic_utils.init_latent(latent, model, height, width, batch_size)
    model.scheduler.set_timesteps(num_inference_steps)

    all_latent_masks = {}
    for key in all_masks.keys():
        all_latent_masks[key] = change_all_masks_shape(all_masks[key], latents)
    
    # TODO There may be ambiguity in using prompt_change as key!
    origin_prompt, guided_prompts, target_prompt = prompts[0], prompts[1:-1], prompts[-1]
    all_embeddings = {}
    for prompt_change, guided_prompt in zip(prompt_changes, guided_prompts):
        all_embeddings[prompt_change] = model.tokenizer(
            [origin_prompt, guided_prompt],
            padding='max_length',
            max_length=model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        all_embeddings[prompt_change] = model.text_encoder(all_embeddings[prompt_change].input_ids.to(model.device))[0]
    target_embedding = model.tokenizer(
        [origin_prompt, target_prompt],
        padding='max_length',
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    target_embedding = model.text_encoder(target_embedding.input_ids.to(model.device))[0]
    
    origin_embedding = model.tokenizer(
        [origin_prompt],
        padding='max_length',
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    origin_embedding = model.text_encoder(origin_embedding.input_ids.to(model.device))[0]
    
    guided_latents = {}
    for prompt_change in prompt_changes:
        context = all_embeddings[prompt_change]
        latent = all_latents[optimal_inversion_steps[prompt_change]]
        timesteps = model.scheduler.timesteps[-(optimal_inversion_steps[prompt_change]):]
        stop_for_reassembly = len(timesteps) - reassembly_step
        for i, t in enumerate(tqdm(timesteps)):
            latent = basic_utils.diffusion_step(model, latent, context, t, guidance_scale, low_resource=False)
            if i == stop_for_reassembly - 1:
                guided_latents[prompt_change] = latent
                break
    
    # crop editing region and non-editing region, and use them to contruct reassembly latent
    reassembly_latent = all_latent_masks['non_editing_region_mask'] * all_latents[reassembly_step]
    for prompt_change in prompt_changes:
        reassembly_latent += all_latent_masks[prompt_change] * guided_latents[prompt_change]        
    
    # use re-inversion and change prompt to target prompt to guided denoising process.
    reassembly_latent = reinversion_and_denoise(
        model,
        reassembly_latent,
        target_embedding,
        origin_embedding,
        ddim_inversion,
        reinversion_steps,
        reassembly_step,
        guidance_scale,
    )
    image = basic_utils.latent2image(model.vae, reassembly_latent) if return_type == 'image' else latents
    return image, reassembly_latent

def reinversion_and_denoise(
        model,
        reassembly_latent,
        target_embedding,
        origin_embedding,
        ddim_inversion,
        reinversion_steps,
        reassembly_step,
        guidance_scale,
):
    reassembly_latent = ddim_inversion.reinversion(reassembly_latent, origin_embedding, reassembly_step, reinversion_steps)
    for t in model.scheduler.timesteps[-(reinversion_steps + reassembly_step):]:
        reassembly_latent = basic_utils.diffusion_step(model, reassembly_latent, target_embedding, t, guidance_scale, low_resource=False)
    return reassembly_latent



