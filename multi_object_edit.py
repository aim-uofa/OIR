import sys, torch, os, cv2, yaml, shutil, argparse, time

import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from configs.oir_config import (
    ldm_stable, 
    NUM_DDIM_STEPS,
    GUIDANCE_SCALE,
    clip_text_path,
)
from utils.basic_utils import change_images_to_file
from utils.candidate_images_generation import candidate_images_generation
from utils.optimal_candidate_selection import optimal_candidate_selection
from utils.oir import oir

from sampler.ddim_inversion import DDIMInversion
sys.path.append("/home/yangzhen/code/DynamicInversion")



def main(args):

    # 0. the basic information of user inputs
    image_path = args['image_path']
    generation_image_path = 'results'
    origin_prompt = args['origin_prompt']
    target_prompt = args['target_prompt']
    guided_prompts = args['guided_prompt']
    origin_changes = args['origin_change']
    prompt_changes = args['prompt_change']
    prompt_changes_mask = args['prompt_change_mask']
    reassembly_step = args['reassembly_step']
    reinversion_steps = args['reinversion_steps']
    
    # 1. Guided prompts preparation
    guided_prompts_list, prompts = [], [origin_prompt]
    for guided_prompt, prompt_change in zip(guided_prompts, prompt_changes):
        guided_prompts_list.append(guided_prompt[0] + prompt_change + guided_prompt[1])
    for prompt in guided_prompts_list:
        prompts.append(prompt)
    prompts.append(target_prompt)

    # 2. inversion
    print('Inversion ...')
    ddim_inversion = DDIMInversion(ldm_stable)
    all_latents = ddim_inversion.invert(image_path, origin_prompt)
    end = time.time()
    
    # 3. collect all candidate images, and save it into the file
    print('Candidate images generation ...')
    # TODO There may be ambiguity in using prompt_change as key!
    candidate_images = {}
    for guided_prompt, prompt_change in zip(guided_prompts_list, prompt_changes):
        candidate_images[prompt_change] = candidate_images_generation(
            ldm_stable, 
            origin_prompt,
            guided_prompt,
            prompt_change,
            num_inference_steps=NUM_DDIM_STEPS, 
            guidance_scale=GUIDANCE_SCALE, 
            all_latents=all_latents,
        )
    
    # 4. select the optimal inversion step from candidate images
    print('Optimal candidate selection ...')
    # TODO There may be ambiguity in using prompt_change as key!
    optimal_inversion_steps, all_masks = {}, {}
    all_masks['non_editing_region_mask'] = 1
    for p_idx, prompt_change, prompt_change_mask in zip(range(len(prompt_change)), prompt_changes, prompt_changes_mask):
        max_idx, _ = optimal_candidate_selection(
            origin_image_path=image_path,
            editing_region_mask_path=prompt_change_mask,
            candidate_images=candidate_images[prompt_change],
            target_prompt_change=prompt_change,
            all_masks=all_masks,
            clip_text_path=clip_text_path,
        )
        optimal_inversion_steps[prompt_changes[p_idx]] = max_idx

    # 5. make sure the optimal inversion steps are arranged from smallest to largest and get all masks
    prompt_changes = sorted(prompt_changes, key=lambda x: optimal_inversion_steps[x])
    all_masks['max_optimal_inversion_step_mask'], all_masks['all_editing_region_mask'] = all_masks[prompt_changes[-1]], 0
    for prompt_change in prompt_changes:
        all_masks['all_editing_region_mask'] += all_masks[prompt_change]

    # 6. implement OIR
    max_optimal_inversion_step = optimal_inversion_steps[prompt_changes[-1]]
    right_to_left_1_point = optimal_inversion_steps[prompt_changes[0]]
    x_t = all_latents[max_optimal_inversion_step]
    images, x_t = oir(
        ldm_stable, 
        prompts, 
        optimal_inversion_steps=optimal_inversion_steps,
        latent=x_t, 
        num_inference_steps=NUM_DDIM_STEPS, 
        guidance_scale=GUIDANCE_SCALE, 

        all_latents=all_latents,
        all_masks=all_masks,

        ddim_inversion=ddim_inversion,
        reinversion_steps=reinversion_steps,

        max_optimal_inversion_step=max_optimal_inversion_step,
        right_to_left_1_point=right_to_left_1_point,
        reassembly_step=reassembly_step,
        prompt_changes=prompt_changes,
    )
    

    Image.fromarray(images.squeeze(0), 'RGB').save('output_image.png')



    
    


if __name__ == '__main__':
    with open('configs/multi_object_edit.yaml', 'r') as file:
        args = yaml.safe_load(file)
    for key in args.keys():
        main(args[key])

# CUDA_VISIBLE_DEVICES=2 python oir_parallel.py --key multi_object_nice_0107

