import torch, os, sys, argparse, time
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from models.mask_clip import MaskCLIPModel
from torchmetrics.multimodal import CLIPScore
from transformers import AutoProcessor, CLIPModel, AutoTokenizer


def change_mask_to_attention_mask(mask, patch_size=16, image_size=224, use_cls_token=True):
    h, w = image_size // patch_size, image_size // patch_size
    seq_len = h * w + 1 if use_cls_token else h * w
    # mask.shape -> (H, W)
    mask = mask.unsqueeze(0).unsqueeze(0)
    # mask.shape -> (N=1, C=1, H, W)
    mask = torch.nn.functional.interpolate(
            mask.float(), 
            size=(h, w),
            mode="nearest",
        ) > 0.5
    # mask.shape -> (N=1, H, W)
    mask = mask.squeeze(1)
    mask = torch.flatten(mask, start_dim=1)
    cls_token = torch.ones_like(mask[:, :1])
    mask = torch.concat((cls_token, mask), dim=1)
    attention_mask = torch.zeros((1, 1, seq_len, seq_len))
    for i in range(seq_len):
        for j in range(seq_len):
            if (mask[:, i] and mask[:, j]):
                attention_mask[:, :, i, j] = 1
    return attention_mask

def change_mask_shape(mask, image):
    _, _, h, w = image.shape
    mask = mask.unsqueeze(0).repeat(1, 3, 1, 1)
    new_mask = torch.nn.functional.interpolate(
        mask.float(), 
        size=(h, w),
        mode="nearest",
    ) > 0.5
    return new_mask


def editing_region_clip_score(
    target_prompt_change,
    candidate_images,
    editing_region_mask,
    clip_text_path,
):
    scores = []
    model = MaskCLIPModel.from_pretrained(clip_text_path)
    processor = AutoProcessor.from_pretrained(clip_text_path)
    tokenizer = AutoTokenizer.from_pretrained(clip_text_path)
    target_prompt_change = tokenizer(target_prompt_change, padding=True, return_tensors="pt")
    target_prompt_change_feature = model.get_text_features(**target_prompt_change)
    if clip_text_path.split('/')[-1] == "clip-vit-base-patch16":
        editing_region_self_attention_mask = change_mask_to_attention_mask(editing_region_mask)
    elif clip_text_path.split('/')[-1] == "clip-vit-large-patch14":
        editing_region_self_attention_mask = change_mask_to_attention_mask(editing_region_mask, patch_size=14)
        
    for i in range(len(candidate_images)):
        target_image = processor(images=candidate_images[i], return_tensors="pt")
        editing_region_of_target_image_feature = model.get_mask_image_features(**target_image, image_attention_mask=editing_region_self_attention_mask)
        score = torch.cosine_similarity(target_prompt_change_feature, editing_region_of_target_image_feature)
        scores.append(score.item())
    return scores

def non_editing_region_negative_MSE(
    origin_image,
    editing_region_mask,
    candidate_images,
    clip_text_path,
):
    scores = []
    processor = AutoProcessor.from_pretrained(clip_text_path)
    origin_image = processor(images=origin_image, return_tensors="pt")['pixel_values']
    non_editing_region_mask = change_mask_shape(1 - editing_region_mask, origin_image)
    non_editing_region_of_origin_image = origin_image * non_editing_region_mask
    for i in range(len(candidate_images)):
        target_image = processor(images=candidate_images[i], return_tensors='pt')['pixel_values']
        non_editing_region_of_target_image = target_image * non_editing_region_mask
        score = -F.mse_loss(non_editing_region_of_origin_image, non_editing_region_of_target_image)
        scores.append(score.item())
    return scores

def optimal_candidate_selection(
    origin_image_path,
    target_prompt_change,
    editing_region_mask_path,
    candidate_images,
    all_masks,
    clip_text_path,
):
    origin_image = np.array(Image.open(origin_image_path))
    editing_region_mask = torch.from_numpy(np.where(np.array(Image.open(editing_region_mask_path)) >= 1, 1, 0))
    # default setting: different editing pairs' masks don't have overlap
    all_masks[target_prompt_change] = editing_region_mask.unsqueeze(0)
    all_masks['non_editing_region_mask'] -= editing_region_mask.unsqueeze(0)
    
    non_editing_region_scores = non_editing_region_negative_MSE(
        origin_image,
        editing_region_mask,
        candidate_images,
        clip_text_path,
    )
    editing_region_scores = editing_region_clip_score(
        target_prompt_change,
        candidate_images,
        editing_region_mask,
        clip_text_path,
    )
    max_non_editing_region_score, min_non_editing_region_score = max(non_editing_region_scores), min(non_editing_region_scores)
    max_editing_region_score, min_editing_region_score = max(editing_region_scores), min(editing_region_scores)
    
    normalize_non_editing_region_scores = [(score - min_non_editing_region_score) / (max_non_editing_region_score - min_non_editing_region_score) for score in non_editing_region_scores]
    normalize_editing_region_scores = [(score - min_editing_region_score) / (max_editing_region_score - min_editing_region_score) for score in editing_region_scores]
    
    search_metric = [normalize_non_editing_region_scores[i] + normalize_editing_region_scores[i] for i in range(len(normalize_editing_region_scores))]
    max_search_metric_idx = search_metric.index(max(search_metric))
    return max_search_metric_idx + 1, candidate_images[max_search_metric_idx]
    
    
