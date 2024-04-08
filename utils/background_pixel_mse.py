import torch
import os
from PIL import Image
import sys
sys.path.append("/home/yangzhen/code/DynamicInversion")
from torchmetrics.multimodal import CLIPScore
from transformers import AutoProcessor, CLIPModel, AutoTokenizer
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from grounded_sam.grounded_sam import generate_mask, generate_boundingbox, change_mask_to_attention_mask, change_mask_to_gray_image
from models.mask_clip import MaskCLIPModel
import argparse
import torch.nn.functional as F
import time


def change_mask_shape(mask, image):
    """
        主要用于mse场景下的图像和mask相乘仅保留background区域
    """
    _, _, h, w = image.shape
    mask = mask.unsqueeze(0).repeat(1, 3, 1, 1)
    new_mask = torch.nn.functional.interpolate(
        mask.float(), 
        size=(h, w),  # TODO: FIXED here
        mode="nearest",
    ) > 0.5
    return new_mask


def background_pixel_mse(
        origin_image_path, 
        root_path,
        classes,
        model_path="/home/yangzhen/checkpoints/openai/clip-vit-base-patch16",
        mask_method='detection', # segmentation
        box_type='global_box', # local box 仅仅针对detection,
        
):
    """
        mask_method: detection, segmentation
        box_type: global_box, local_box

        计算origin image background和target image background之间的MSE
        可以控制是否使用attention mask, 该attention mask会对ViT的所有层都使用，防止信息泄漏
        mask method控制那种mask方案，一种是矩形框形式，会让物体有变化的自由度，另一种是完全的segmentation，这种形式对物体形状限制很死
        box_type仅仅对detection有用，global box代表用一个矩形框住所有物体，local box代表用不规则矩形框住所有物体
    """    
    model = MaskCLIPModel.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(model_path)
    origin_image = Image.open(origin_image_path)
    inputs_of_origin_image = processor(images=origin_image, return_tensors="pt")

    if mask_method == 'detection':
        object_mask = generate_boundingbox(
            origin_image_path,
            classes,
            model.device,
            save_image=True, 
            output_path='/home/yangzhen/code/DynamicInversion/outputs/GroundedSAM/detection_background_pixel_mse_decrease.png',
            box_type=box_type,
        )
    elif mask_method == 'segmentation':
        object_mask = generate_mask(
            origin_image_path,
            classes,
            model.device,
            save_image=True, 
            output_path='/home/yangzhen/code/DynamicInversion/outputs/GroundedSAM/segmentation_background_pixel_mse_decrease.png'
        )
        

    background_mask = change_mask_shape(1 - object_mask, inputs_of_origin_image['pixel_values'])
    pixel_of_origin_background = inputs_of_origin_image['pixel_values'] * background_mask

    scores = []
    print('background_pixel_mse_decrease')
    
    start = time.time()
    for i in tqdm(range(1, 1000)):
        img_path = os.path.join(root_path, str(i).zfill(4) + '.png')
        if not os.path.exists(img_path):
            continue
        target_image = Image.open(img_path)
        inputs_of_target_image = processor(images=target_image, return_tensors="pt")
        pixel_of_target_background = inputs_of_target_image['pixel_values'] * background_mask
        score = F.mse_loss(pixel_of_origin_background, pixel_of_target_background)
        scores.append(score.item())
    end = time.time()
    print("\n\nMSE: ", end - start, '\n\n')
    return scores
