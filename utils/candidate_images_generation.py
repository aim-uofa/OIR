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
import time
from tqdm import trange
import os

@torch.no_grad()
def candidate_images_generation(
    model,
    origin_prompt,
    guided_prompt,
    prompt_change,
    all_latents=[],
    num_inference_steps: int = 50,
    guidance_scale: Optional[float] = 7.5,
    height=512,
    width=512,
    save_path='',
):
    origin_image_latent = all_latents[0]
    batch_size = num_inference_steps // 2
    text_inputs = model.tokenizer(
        [origin_prompt, guided_prompt],
        padding="max_length",
        max_length=model.tokenizer.model_max_length,
        truncation=True,
        return_tensors="pt",
    )
    max_length = text_inputs.input_ids.shape[-1]
    text_embeddings = model.text_encoder(text_inputs.input_ids.to(model.device))[0]
    original_context = text_embeddings[0].unsqueeze(0).repeat(batch_size, 1, 1)

    origin_image_latent, latents = basic_utils.init_latent_parallel(
        origin_image_latent, 
        model, height, width, 
        batch_size, 
        all_latents, 
        num_inference_steps,
    )
    
    model.scheduler.set_timesteps(num_inference_steps)
    temporal_timesteps = torch.cat([model.scheduler.timesteps[-1].unsqueeze(0), model.scheduler.timesteps]).unsqueeze(0).repeat(batch_size, 1)
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

    text_idx = 1
    context = torch.cat([original_context, text_embeddings[text_idx].unsqueeze(0).repeat(num_inference_steps // 2, 1, 1)])
    edited_latents = None
    for i in range(num_inference_steps + 1):
        t = temporal_timesteps[:, i]
        latents = basic_utils.diffusion_step_parallel(
            model, latents, context, t, 
            guidance_scale, 
            low_resource=False,
            use_parallel=True,
        )
        if i < num_inference_steps // 2:
            if edited_latents is None:
                edited_latents = latents[i].unsqueeze(0)
            else:
                edited_latents = torch.cat([edited_latents, latents[i].unsqueeze(0)])
            latents[i] = all_latents[-1 - i]
    
    for i in reversed(range(latents.shape[0])):
        edited_latents = torch.cat([edited_latents, latents[i].unsqueeze(0)])
    edited_latents[0] = all_latents[0]
    images = basic_utils.latent2image(model.vae, edited_latents)
    # basic_utils.view_images(images, save_path=os.path.join(save_path, prompt_change[text_idx - 1]) + '/', file_name="0000.png")    
    return images