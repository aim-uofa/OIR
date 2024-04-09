import sys, torch, os, cv2, yaml, shutil, argparse, time

import numpy as np
from PIL import Image
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from configs.oir_config import (
    ldm_stable, 
    mask_at_where,
    unmask_area_recover_method,
    normalize_method,
    
    use_reinversion,
    reassembly_step,
    blended_all_step_after_reassembly_step,

    NUM_DDIM_STEPS,
    GUIDANCE_SCALE,
    args,
    clip_text_path,
)
from utils.basic_utils import change_images_to_file
from utils.candidate_images_generation import candidate_images_generation
from utils.optimal_candidate_selection import optimal_candidate_selection
from utils.oir import oir

from sampler.ddim_inversion import DDIMInversion
from sampler.oir_denoise import dynamic_run_and_display
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
    guided_prompts_list = []
    for guided_prompt, prompt_change in zip(guided_prompts, prompt_changes):
        guided_prompts_list.append(guided_prompt[0] + prompt_change + guided_prompt[1])
    

    # 2. inversion
    print('Inversion ...')
    ddim_inversion = DDIMInversion(ldm_stable)
    all_latents = ddim_inversion.invert(image_path, origin_prompt)
    end = time.time()
    
    # 3. collect all candidate images, and save it into the file
    print('Candidate images generation ...')
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

    # 6. OIR
    latent_max = all_latents[prompt_changes[-1]]
    images, x_t = oir(
        
    )
    

    # # 6. 生成local mask
    # image_mask = {}
    # image_mask['background'] = 1
    # if not os.path.exists(generation_image_path_ldi_mask):
    #     os.makedirs(generation_image_path_ldi_mask)
    # for p_o, p_t in zip(origin_changes, prompt_changes):
    #     mask = generate_mask(
    #         image_path,
    #         [p_o,],
    #         ldm_stable.device,
    #         save_image=True,
    #         output_path=generation_image_path_ldi_mask + p_o + '.png',
    #     )
    #     image_mask[p_t] = mask
    #     image_mask['background'] = image_mask['background'] - mask

    # if len(prompt_changes) == 2:
    #     # 保证从左往右inverison point逐渐变大
    #     if sweet_results[prompt_changes[0]] > sweet_results[prompt_changes[1]]:
    #         prompts[1], prompts[2] = prompts[2], prompts[1]
    #         origin_changes[0], origin_changes[1] = origin_changes[1], origin_changes[0]
    #         prompt_changes[0], prompt_changes[1] = prompt_changes[1], prompt_changes[0]
    # else:
    #     assert('object greater than 2')

    # image_mask['right_to_left_1_point'], image_mask['reassembly_step'] = image_mask[prompt_changes[1]], image_mask[prompt_changes[0]] + image_mask[prompt_changes[1]]
    # max_sweet_point = sweet_results[prompt_changes[1]] # gift
    # right_to_left_1_point = sweet_results[prompt_changes[0]] # dog


    # 5. 生成进行ldi
    x_t = all_latents[max_sweet_point]
    images, x_t = oir(
        ldm_stable, 
        prompts, 
        latent=latent, 
        num_inference_steps=NUM_DDIM_STEPS, 
        guidance_scale=GUIDANCE_SCALE, 

        all_latents=all_latents,
        image_mask=image_mask,

        ddim_inversion=ddim_inversion,
        use_reinversion=use_reinversion,
        reinversion_steps=reinversion_steps,

        max_sweet_point=max_sweet_point,
        right_to_left_1_point=right_to_left_1_point,
        reassembly_step=reassembly_step,
        prompt_changes=prompt_changes,
    )
    
    
    
    start = time.time()
    images, _ = dynamic_run_and_display(
        prompts, 
        run_baseline=False, 
        latent=x_t, 
        uncond_embeddings=None,
        save_path=generation_image_path_ldi,
        file_name='oir_output.png',
        use_negative_prompt_inversion=use_negative_prompt_inversion,

        method='local dynamic inversion',
        all_latents=all_latents,

        image_mask=image_mask,
        mask_at_where=mask_at_where,
        unmask_area_recover_method=unmask_area_recover_method,

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
    end = time.time()
    print('\n\n\nOIR: ', end - start)
    


if __name__ == '__main__':

    main(args)

# CUDA_VISIBLE_DEVICES=2 python oir_parallel.py --key multi_object_nice_0107

