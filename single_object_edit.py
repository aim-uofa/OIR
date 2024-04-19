import sys, torch, os, cv2, yaml, shutil, argparse, time, yaml

import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from configs.basic_config import (
    ldm_stable, 
    NUM_DDIM_STEPS,
    GUIDANCE_SCALE,
    clip_text_path,
)

from utils.candidate_images_generation import candidate_images_generation
from utils.optimal_candidate_selection import optimal_candidate_selection

from sampler.ddim_inversion import DDIMInversion
from sampler.oir_denoise import dynamic_run_and_display



def main(args):
    
    # 0. the basic information of user inputs
    image_path = args['image_path']
    generation_image_path = args['generation_image_path']
    origin_prompt = args['origin_prompt']
    target_prompt_list = args['target_prompt_list']
    prompt_changes = args['target_changes']
    origin_prompt_mask = args['origin_prompt_mask']
        
    # 1. Target prompts preparation
    target_prompts_list = []
    for prompt_change in prompt_changes:
        target_prompts_list.append(target_prompt_list[0] + prompt_change + target_prompt_list[1])

    # 2. inversion
    print('Inversion ...')
    ddim_inversion = DDIMInversion(ldm_stable)
    all_latents = ddim_inversion.invert(image_path, origin_prompt)
    end = time.time()
    
    # 3. collect all candidate images
    print('Candidate images generation ...')
    candidate_images = {}
    for target_prompt, prompt_change in zip(target_prompts_list, prompt_changes):
        candidate_images[prompt_change] = candidate_images_generation(
            ldm_stable, 
            origin_prompt,
            target_prompt,
            prompt_change,
            num_inference_steps=NUM_DDIM_STEPS, 
            guidance_scale=GUIDANCE_SCALE, 
            all_latents=all_latents,
        )
    
    # 4. select the optimal inversion step from candidate images
    print('Optimal candidate selection ...')
    optimal_inversion_steps, all_masks = {}, {}
    all_masks['non_editing_region_mask'] = 1
    for p_idx, prompt_change in zip(range(len(prompt_change)), prompt_changes):
        max_idx, output_image = optimal_candidate_selection(
            origin_image_path=image_path,
            editing_region_mask_path=origin_prompt_mask,
            candidate_images=candidate_images[prompt_change],
            target_prompt_change=prompt_change,
            all_masks=all_masks,
            clip_text_path=clip_text_path,
        )
        optimal_inversion_steps[prompt_changes[p_idx]] = max_idx
        if not os.path.exists(generation_image_path):
            os.makedirs(generation_image_path)    
        img = Image.fromarray(output_image).save(os.path.join(generation_image_path, prompt_change + '.png'))

if __name__ == '__main__':
    with open('configs/single_object_edit.yaml', 'r') as file:
        args = yaml.safe_load(file)
    for key in args.keys():
        main(args[key])

