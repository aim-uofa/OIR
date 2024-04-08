

import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch, os, sys, argparse, time
from models.mask_clip import MaskCLIPModel
from torchmetrics.multimodal import CLIPScore
sys.path.append("/home/yangzhen/code/DynamicInversion")
from transformers import AutoProcessor, CLIPModel, AutoTokenizer



from grounded_sam.grounded_sam import generate_mask, generate_boundingbox, change_mask_to_attention_mask, change_mask_to_gray_image




def object_clip_score(
        origin_image_path,  # 用于提取mask
        root_path,
        classes,
        target_prompt,
        model_path="/home/yangzhen/checkpoints/openai/clip-vit-base-patch16",
        use_attention_mask=True,
        mask_method='detection',
        box_type='global box',
        
        
):
    """
        用来计算target prompt和target image object区域的相似程度
    
    """
    model = MaskCLIPModel.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(target_prompt, padding=True, return_tensors="pt")
    feature_of_origin_prompt = model.get_text_features(**inputs)

    if mask_method == 'detection':
        object_mask = generate_boundingbox(
            origin_image_path,
            classes,
            model.device,
            save_image=True, 
            output_path='/home/yangzhen/code/DynamicInversion/outputs/GroundedSAM/detection_object_clip_score.png',
            box_type=box_type,
        )
    elif mask_method == 'segmentation':
        object_mask = generate_mask(
            origin_image_path,
            classes,
            model.device,
            save_image=True, 
            output_path='/home/yangzhen/code/DynamicInversion/outputs/GroundedSAM/segmentation_object_clip_score.png'
        )
        
        
    if model_path.split('/')[-1] == "clip-vit-base-patch16":
        object_attention_mask = change_mask_to_attention_mask(object_mask) if use_attention_mask else None
    elif model_path.split('/')[-1] == "clip-vit-large-patch14":
        object_attention_mask = change_mask_to_attention_mask(object_mask, patch_size=14) if use_attention_mask else None

    scores = []
    print('generate similiarity of target image out of editing zone and origin image out of editing zone')
    total_time = 0
    
    for i in tqdm(range(1, 1000)):
        img_path = os.path.join(root_path, str(i).zfill(4) + '.png')
        if not os.path.exists(img_path):
            continue
        target_image = Image.open(img_path)
        inputs_of_target_image = processor(images=target_image, return_tensors="pt")
        start = time.time()
        feature_of_target_image = model.get_mask_image_features(**inputs_of_target_image, image_attention_mask=object_attention_mask)
        score = torch.cosine_similarity(feature_of_origin_prompt, feature_of_target_image)
        end = time.time()
        total_time += end - start
        scores.append(score.item())
    
    print("\n\nCLIP Score: ", total_time, '\n\n')
    return scores

def object_clip_score_parallel(
        origin_image_path,  # 用于提取mask
        root_path,
        classes,
        target_prompt,
        model_path="/home/yangzhen/checkpoints/openai/clip-vit-base-patch16",
        # model_path="/home/yangzhen/checkpoints/huggingface/models/CLIP/clip-vit-large-patch14",
        use_attention_mask=True,

        mask_method='detection', # segmentation
        box_type='global box', # local box 仅仅针对detection
        

):
    """
        用来计算target prompt和target image object区域的相似程度
    
    """
    model = MaskCLIPModel.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    inputs = tokenizer(target_prompt, padding=True, return_tensors="pt")
    feature_of_origin_prompt = model.get_text_features(**inputs)

    object_mask = generate_mask(
        origin_image_path,
        classes,
        model.device,
        save_image=True, 
        output_path='/home/yangzhen/code/DynamicInversion/outputs/GroundedSAM/segmentation_object_clip_score.png'
    )
    

    if model_path.split('/')[-1] == "clip-vit-base-patch16":
        object_attention_mask = change_mask_to_attention_mask(object_mask) if use_attention_mask else None
    elif model_path.split('/')[-1] == "clip-vit-large-patch14":
        object_attention_mask = change_mask_to_attention_mask(object_mask, patch_size=14) if use_attention_mask else None

    scores = []
    print('generate similiarity of target image out of editing zone and origin image out of editing zone')
    
    all_images = None
    for i in tqdm(range(1, 1000)):
        img_path = os.path.join(root_path, str(i).zfill(4) + '.png')
        if not os.path.exists(img_path):
            continue
        target_image = Image.open(img_path)
        inputs_of_target_image = processor(images=target_image, return_tensors="pt")
        if all_images is None:
            all_images = inputs_of_target_image
        else:
            all_images['pixel_values'] = torch.concat([all_images['pixel_values'], inputs_of_target_image['pixel_values']])
    start = time.time()
    feature_of_target_image = model.get_mask_image_features(**all_images, image_attention_mask=object_attention_mask.repeat(all_images['pixel_values'].shape[0], 1, 1, 1))
    scores = torch.cosine_similarity(feature_of_origin_prompt, feature_of_target_image)
    end = time.time()
    print("\n\nCLIP Score: ", end - start, '\n\n')
    return scores.tolist()

