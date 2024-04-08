import sys
sys.path.append("/home/yangzhen/code/DynamicInversion")
from tqdm.notebook import tqdm
import torch
import os
import cv2
import yaml
import shutil
from configs.oir_config import (
    ldm_stable, 
    use_negative_prompt_inversion,
    mask_at_where,
    unmask_area_recover_method,
    normalize_method,

    use_reinversion,
    reassembly_step,
    blended_all_step_after_background_sweet_point,
    background_mask_method,
    background_box_type,
    object_mask_method,
    object_box_type,

    NUM_DDIM_STEPS,
    GUIDANCE_SCALE,
    args,
)
from utils.basic_utils import change_images_to_file
from sampler.ddim_inversion import DDIMInversion
from sampler.oir_denoise import dynamic_run_and_display
from sampler.oir_denoise import collect_candidate_images
from grounded_sam.grounded_sam import (
    generate_mask, 
    generate_boundingbox,
    change_mask_to_self_attention_mask,
    change_mask_to_cross_attention_mask,
)
import matplotlib.pyplot as plt
from utils.visualization_object_background import synthesize_object_background, synthesize_object_background_loadtxt
import argparse
import time


def main(args):

    # 0. the basic information of user inputs
    image_path = args['image_path']
    generation_image_path = 'results'
    origin_prompt = args['origin_prompt']
    target_prompt = args['target_prompt']
    guided_prompts = args['guided_prompt']
    origin_changes = args['origin_change']
    prompt_changes = args['prompt_change']
    reassembly_step = args['reassembly_step']
    generation_image_path_gdi = args['generation_image_path_gdi']
    generation_image_path_ldi = args['generation_image_path_ldi']
    generation_image_path_ldi_mask = args['generation_image_path_ldi_mask']
    reinversion_steps = args['reinversion_steps']
    
    # 1. initialize prompt's information
    ddim_inversion = DDIMInversion(ldm_stable)
    prompts = [origin_prompt]
    for guided_prompt, prompt_change in zip(guided_prompts, prompt_changes):
        prompts.append(guided_prompt[0] + prompt_change + guided_prompt[1])
    prompts.append(target_prompt)

    # 2. inversion
    print('Inversion ...')
    all_latents = ddim_inversion.invert(image_path, origin_prompt)
    end = time.time()
    
    # 3. collect all candidate images, and save it into the file
    print('Collect candidate images ...')
    collect_candidate_images(
        ldm_stable, 
        prompts, 
        latent=all_latents[0], 
        num_inference_steps=NUM_DDIM_STEPS, 
        guidance_scale=GUIDANCE_SCALE, 
        use_negative_prompt_inversion=use_negative_prompt_inversion,
        all_latents=all_latents,
        prompt_changes=prompt_changes,
        save_path=generation_image_path_gdi,
    )

    # 4. crop the image and save it for easy visualization
    for i in range(len(prompt_changes)):
        change_images_to_file(
            generated_images_path=os.path.join(generation_image_path_gdi, prompt_changes[i]),
            image_name='0000.png',
            num_steps=NUM_DDIM_STEPS,
        )
    
    # 5. use search metric to find and visualize optimal inversion step
    metric_path = os.path.join(generation_image_path_gdi, 'metric')
    sweet_result_path = os.path.join(generation_image_path_gdi, 'sweet_result')
    if not os.path.exists(metric_path):
        os.makedirs(metric_path)
    if not os.path.exists(sweet_result_path):
        os.makedirs(sweet_result_path)

    background_score_save_path = [os.path.join(generation_image_path_gdi, 'metric', prompt + '_decrease_metric.txt') for prompt in prompt_changes]
    object_score_save_path = [os.path.join(generation_image_path_gdi, 'metric', prompt + '_increase_metric.txt') for prompt in prompt_changes]        
    plt.figure(figsize=(60, 30))
    colors = ['r-o', 'm-o',]
    root_paths = [os.path.join(generation_image_path_gdi, prompt) for prompt in prompt_changes]
    legend_list = prompt_changes
    for p in range(len(origin_changes)):
        synthesize_object_background(
            origin_image_path=image_path,
            root_path=root_paths[p],
            classes=[origin_changes[p]],
            target_prompt=prompt_changes[p],
            object_metric='object_clip_score_parallel', # object_clip_score_parallel/object_clip_score
            background_metric='background_pixel_mse', # background_clip_score, background_pixel_mse
            use_attention_mask=True,
            title='synthesize_object_background', 
            y_label='score',
            legend_list=legend_list,
            display_separately=False,
            save_path=os.path.join(generation_image_path_gdi, 'metric/synthesize_object_background.png'),
            color=colors[p],
            background_score_save_path=background_score_save_path[p],
            object_score_save_path=object_score_save_path[p],
            normalize_method=normalize_method,
            background_mask_method=background_mask_method,
            background_box_type=background_box_type,
            object_mask_method=object_mask_method,
            object_box_type=object_box_type,
        )
    plt.figure(figsize=(60, 30))
    sweet_results = {}
    
    for p in range(len(prompt_changes)):
        max_idx = synthesize_object_background_loadtxt(
            background_score_save_path[p],
            object_score_save_path[p],
            colors[p],
            normalization_method=normalize_method,
            y_label='S',
            title='Search Metric',
            legend_list=legend_list,
            save_path=os.path.join(generation_image_path_gdi, 'metric/synthesize_object_background_loadtxt.png'),
        )
        
        oldname = os.path.join(generation_image_path_gdi, prompt_changes[p], str(max_idx).zfill(4) + '.png')
        newname = os.path.join(generation_image_path_gdi, 'sweet_result', prompt_changes[p] + '_' + str(max_idx).zfill(4) + '.png')
        shutil.copyfile(oldname, newname)
        sweet_results[prompt_changes[p]] = max_idx


########################################################################################################################
    # 6. 生成local mask
    image_mask = {}
    image_mask['background'] = 1
    if not os.path.exists(generation_image_path_ldi_mask):
        os.makedirs(generation_image_path_ldi_mask)
    for p_o, p_t in zip(origin_changes, prompt_changes):
        mask = generate_mask(
            image_path,
            [p_o,],
            ldm_stable.device,
            save_image=True,
            output_path=generation_image_path_ldi_mask + p_o + '.png',
        )
        image_mask[p_t] = mask
        image_mask['background'] = image_mask['background'] - mask

    if len(prompt_changes) == 2:
        # 保证从左往右inverison point逐渐变大
        if sweet_results[prompt_changes[0]] > sweet_results[prompt_changes[1]]:
            prompts[1], prompts[2] = prompts[2], prompts[1]
            origin_changes[0], origin_changes[1] = origin_changes[1], origin_changes[0]
            prompt_changes[0], prompt_changes[1] = prompt_changes[1], prompt_changes[0]
    else:
        assert('object greater than 2')

    image_mask['right_to_left_1_point'], image_mask['reassembly_step'] = image_mask[prompt_changes[1]], image_mask[prompt_changes[0]] + image_mask[prompt_changes[1]]
    max_sweet_point = sweet_results[prompt_changes[1]] # gift
    right_to_left_1_point = sweet_results[prompt_changes[0]] # dog
    replace_all_unmask_area_after_replace_point = False


    # 5. 生成进行ldi
    x_t = all_latents[max_sweet_point]
    
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
        blended_all_step_after_background_sweet_point=blended_all_step_after_background_sweet_point,
        prompt_changes=prompt_changes,
    )
    end = time.time()
    print('\n\n\nOIR: ', end - start)
    


if __name__ == '__main__':

    main(args)

# CUDA_VISIBLE_DEVICES=2 python oir_parallel.py --key multi_object_nice_0107

