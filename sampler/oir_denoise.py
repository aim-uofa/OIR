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
def collect_candidate_images(
    model,
    prompt:  List[str],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator = None, # generator: Optional[torch.Genwerator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    return_type='image',

    use_negative_prompt_inversion=True,
    
    all_latents=[],
    prompt_changes=[],
    save_path='',
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
        
    batch_size = NUM_DDIM_STEPS // 2
    latent, latents = basic_utils.init_latent_parallel(latent, model, height, width, generator, batch_size, all_latents, NUM_DDIM_STEPS)
    original_context = text_embeddings[0].unsqueeze(0).repeat(batch_size, 1, 1)
    
    model.scheduler.set_timesteps(num_inference_steps)
    temporal_timesteps = torch.cat([model.scheduler.timesteps[-1].unsqueeze(0), model.scheduler.timesteps])
    temporal_timesteps = temporal_timesteps.unsqueeze(0).repeat(NUM_DDIM_STEPS // 2, 1)
    """
       [[  1, 981, 961,  ...,  41,  21,   1],
        [ 21,   1, 961,  ...,  41,  21,   1],
        [ 41,  21,   1,  ...,  41,  21,   1],
        ...,
        [441, 421, 401,  ...,  41,  21,   1],
        [461, 441, 421,  ...,  41,  21,   1],
        [481, 461, 441,  ...,  41,  21,   1]]
    """
    for m in range(1, temporal_timesteps.shape[0]):
        for n in range(temporal_timesteps.shape[1]):
            if n <= m:
                temporal_timesteps[m][n] = model.scheduler.timesteps[n - m - 1]

    inital_latent = latents
    for text_idx in trange(1, text_embeddings.shape[0] - 1):
        latents = inital_latent
        context = torch.cat([original_context, text_embeddings[text_idx].unsqueeze(0).repeat(NUM_DDIM_STEPS // 2, 1, 1)])
        edited_latents = None
        for i in range(NUM_DDIM_STEPS + 1):
            t = temporal_timesteps[:, i]

            latents = basic_utils.diffusion_step_parallel(
                model, 
                latents, 
                context, 
                t, 
                guidance_scale, 
                low_resource=False,
                use_parallel=True,
            )
            
            if i < NUM_DDIM_STEPS // 2:
                if edited_latents is None:
                    edited_latents = latents[i].unsqueeze(0)
                else:
                    edited_latents = torch.cat([edited_latents, latents[i].unsqueeze(0)])
                latents[i] = all_latents[-1 - i]
        
        for i in reversed(range(latents.shape[0])):
            edited_latents = torch.cat([edited_latents, latents[i].unsqueeze(0)])
        edited_latents[0] = all_latents[0]
        images = basic_utils.latent2image(model.vae, edited_latents)
        basic_utils.view_images(images, save_path=os.path.join(save_path, prompt_changes[text_idx - 1]) + '/', file_name="0000.png")    
    return images, latent


@torch.no_grad()
def dynamic_text2image_ldm_stable_local_dynamic_inversion(
    model,
    prompt:  List[str],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    generator = None, # generator: Optional[torch.Genwerator] = None,
    latent: Optional[torch.FloatTensor] = None,
    uncond_embeddings=None,
    return_type='image',

    use_negative_prompt_inversion=True,

    unet_self_attention_mask=None,
    unet_cross_attention_mask=None,
    position_to_use_mask=[],

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
    blended_all_step_after_background_sweet_point=False,

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

        latents = basic_utils.diffusion_step(
            model, 
            latents, context, t, guidance_scale, low_resource=False,
            self_attention_mask=unet_self_attention_mask,
            cross_attention_mask=unet_cross_attention_mask,
            position_to_use_mask=position_to_use_mask,
            latent_mask=latent_mask,
            use_cfg_mask=use_cfg_mask,
        )


        if i == right_to_left_1_point_area_latent_save_point_denoise_axis:
            latents[1, :, :, :] = all_latents[right_to_left_1_point]

        if i == reassembly_step_area_latent_save_point_denoise_axis or (blended_all_step_after_background_sweet_point and i > reassembly_step_area_latent_save_point_denoise_axis):
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
                    unet_self_attention_mask,
                    guidance_scale,
                    unet_cross_attention_mask,
                    position_to_use_mask,
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
        unet_self_attention_mask,
        guidance_scale,
        unet_cross_attention_mask,
        position_to_use_mask,
        latent_mask,
        use_cfg_mask,
):
    latents = ddim_inversion.reinversion(latents, unmask_area_latent_save_point, reinversion_steps)
    for t in model.scheduler.timesteps[-(reinversion_steps + unmask_area_latent_save_point): -unmask_area_latent_save_point]:
        latents = basic_utils.diffusion_step(
            model, 
            latents, context, t, guidance_scale, low_resource=False,
            self_attention_mask=unet_self_attention_mask,
            cross_attention_mask=unet_cross_attention_mask,
            position_to_use_mask=position_to_use_mask,
            latent_mask=latent_mask,
            use_cfg_mask=use_cfg_mask,
        )
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

    unet_self_attention_mask=None,
    unet_cross_attention_mask=None,
    position_to_use_mask=[],

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
    blended_all_step_after_background_sweet_point=False, 
    prompt_changes=[]
):
    if method == 'local dynamic inversion':
        images, x_t = dynamic_text2image_ldm_stable_local_dynamic_inversion(
            ldm_stable, 
            prompts, 
            latent=latent, 
            num_inference_steps=NUM_DDIM_STEPS, 
            guidance_scale=GUIDANCE_SCALE, 
            generator=generator, 
            uncond_embeddings=uncond_embeddings,
            use_negative_prompt_inversion=use_negative_prompt_inversion,

            unet_self_attention_mask=unet_self_attention_mask,
            unet_cross_attention_mask=unet_cross_attention_mask,
            position_to_use_mask=position_to_use_mask,

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
            blended_all_step_after_background_sweet_point=blended_all_step_after_background_sweet_point,
            prompt_changes=prompt_changes,
        )
    elif method == 'search metric':
        images, x_t = search_metric(
            ldm_stable, 
            prompts, 
            latent=latent, 
            num_inference_steps=NUM_DDIM_STEPS, 
            guidance_scale=GUIDANCE_SCALE, 
            generator=generator, 
            uncond_embeddings=uncond_embeddings,
            use_negative_prompt_inversion=use_negative_prompt_inversion,
            
            all_latents=all_latents,
            prompt_changes=prompt_changes,
            save_path=save_path,
        )
        return images, x_t

    
    if verbose:
        basic_utils.view_images(images, save_path=save_path, file_name=file_name)
    return images, x_t

