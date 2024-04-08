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
import argparse
import torch.nn.functional as F
from models.mask_clip import MaskCLIPModel
from utils.background_pixel_mse import background_pixel_mse
from utils.object_clip_score import object_clip_score, object_clip_score_parallel


LineWidth = 20
legend_frontsize = 120
label_frontsize = 120
tick_frontsize = 100
title_frontsize = 120


def synthesize_object_background(
        origin_image_path,
        root_path,
        classes,
        target_prompt,
        object_metric,
        background_metric,        
        use_attention_mask=True,
        title = 'local metrics',
        y_label = 'score',
        legend_list = ['decrease score', 'increase score'],
        save_path = '/home/yangzhen/code/DynamicInversion/outputs/metrics/decrease_and_increase_score.png',
        color='r-o',
        display_separately=True,
        background_score_save_path=None,
        object_score_save_path=None,

        normalize_method='max min',
        background_box_type='global box',
        background_mask_method='detection',
        object_box_type='global box',
        object_mask_method='detection',
        
):
    if object_metric == 'object_clip_score_parallel':
        object_score = object_clip_score_parallel(
            origin_image_path=origin_image_path,
            root_path=root_path,
            classes=classes,
            target_prompt=target_prompt,
            use_attention_mask=use_attention_mask,

            mask_method=object_mask_method,
            box_type=object_box_type,
        )

    if background_metric == 'background_pixel_mse':
        background_score = background_pixel_mse(
            origin_image_path=origin_image_path,
            root_path=root_path,
            classes=classes,
            model_path="/home/yangzhen/checkpoints/openai/clip-vit-base-patch16",
            box_type=background_box_type,
            mask_method=background_mask_method,
        )
        background_score = [-score for score in background_score]

    
    if background_score_save_path is not None:
        np.savetxt(background_score_save_path, np.array(background_score))
    if object_score_save_path is not None:
        np.savetxt(object_score_save_path, np.array(object_score))

    if normalize_method == 'max-min':
        max_background_score, min_background_score = max(background_score), min(background_score)
        max_object_score, min_object_score = max(object_score), min(object_score)
        object_score = [(score - min_object_score) / (max_object_score - min_object_score) for score in object_score]
    elif normalize_method == 'z-score':
        mean_background_score, std_background_score = np.mean(background_score), np.std(background_score)
        mean_object_score, std_object_score = np.mean(background_score), np.std(background_score)
        background_score = (background_score - mean_background_score) / std_background_score
        object_score = (object_score - mean_object_score) / std_object_score
    else:
        pass

    plt.xticks(fontsize=tick_frontsize)
    plt.yticks(fontsize=tick_frontsize)

    t = list(range(len(background_score)))
    score = [background_score[i] + object_score[i] for i in range(len(background_score))]
    if display_separately:
        plt.plot(t, background_score, color, linewidth=LineWidth)
        plt.plot(t, object_score, color, linewidth=LineWidth)
    else:
        plt.plot(t, score, color, linewidth=LineWidth)

    plt.title(title, fontsize=title_frontsize)
    plt.xlabel("t", fontsize=label_frontsize)
    plt.ylabel(y_label, fontsize=label_frontsize)

    plt.legend(legend_list, fontsize=legend_frontsize)
    plt.savefig(save_path)


def background_score_loadtxt(
    list_background_save_path,
    color,
    normalization_method='max-min',
    y_label='decrease score',
    title='local metrics',
    legend_list=['decrease score', 'increase score'],
    save_path='/home/yangzhen/code/DynamicInversion/outputs/metrics/decrease.png',
):
    decrease_score = np.loadtxt(list_background_save_path)
    if normalization_method == 'max-min':
        max_decrease_score, min_decrease_score = max(decrease_score), min(decrease_score)
        normalize_decrease_score = [(score - min_decrease_score) / (max_decrease_score - min_decrease_score) for score in decrease_score]
    elif normalization_method == 'z-score':
        mean_decrease_score, std_decrease_score = np.mean(decrease_score), np.std(decrease_score)
        normalize_decrease_score = (decrease_score - mean_decrease_score) / std_decrease_score
    else:
        normalize_decrease_score = decrease_score


    plt.xticks(fontsize=tick_frontsize)
    plt.yticks(fontsize=tick_frontsize)

    t = list(range(len(decrease_score)))
    plt.plot(t, normalize_decrease_score, color, linewidth=LineWidth)
    plt.title(title, fontsize=title_frontsize)
    plt.xlabel("t", fontsize=label_frontsize)
    plt.ylabel(y_label, fontsize=label_frontsize)
    plt.legend(legend_list, fontsize=legend_frontsize)
    plt.savefig(save_path)

def object_score_loadtxt(
    list_object_save_path,
    color,
    normalization_method='max-min',
    y_label='increase score',
    title='local metrics',
    legend_list=['decrease score', 'increase score'],
    save_path='/home/yangzhen/code/DynamicInversion/outputs/metrics/increase.png',
):
    increase_score = np.loadtxt(list_object_save_path)
    if normalization_method == 'max-min':
        max_increase_score, min_increase_score = max(increase_score), min(increase_score)
        normalize_decrease_score = [(score - min_increase_score) / (max_increase_score - min_increase_score) for score in increase_score]
    elif normalization_method == 'z-score':
        mean_increase_score, std_increase_score = np.mean(increase_score), np.std(increase_score)
        normalize_decrease_score = (increase_score - mean_increase_score) / std_increase_score
    else:
        normalize_decrease_score = increase_score

    plt.xticks(fontsize=tick_frontsize)
    plt.yticks(fontsize=tick_frontsize)

    t = list(range(len(increase_score)))
    plt.plot(t, normalize_decrease_score, color, linewidth=LineWidth)
    plt.title(title, fontsize=title_frontsize)
    plt.xlabel("t", fontsize=label_frontsize)
    plt.ylabel(y_label, fontsize=label_frontsize)
    plt.legend(legend_list, fontsize=legend_frontsize)
    plt.savefig(save_path)


def synthesize_object_background_loadtxt(
    background_score_save_path,
    object_score_save_path,
    color,
    normalization_method='max-min',
    y_label='score',
    title='local metrics',
    legend_list=['decrease score', 'increase score'],
    save_path='/home/yangzhen/code/DynamicInversion/outputs/metrics/de_and_in_score.png',
):
    decrease_score = np.loadtxt(background_score_save_path)
    increase_score = np.loadtxt(object_score_save_path)
    ratio_of_decrease = 0.5
    ratio_of_increase = 1 - ratio_of_decrease
    if normalization_method == 'max-min':
        max_decrease_score, max_increase_score = max(decrease_score), max(increase_score)
        min_decrease_score, min_increase_score = min(decrease_score), min(increase_score)
        normalize_decrease_score = [(score - min_decrease_score) / (max_decrease_score - min_decrease_score) for score in decrease_score]
        normalize_increase_score = [(score - min_increase_score) / (max_increase_score - min_increase_score) for score in increase_score]
        score = [ratio_of_decrease * normalize_decrease_score[i] + ratio_of_increase * normalize_increase_score[i] for i in range(len(decrease_score))]
    elif normalization_method == 'z-score':
        mean_decrease_score, mean_increase_score = np.mean(decrease_score), np.mean(increase_score)
        std_decrease_score, std_increase_score = np.std(decrease_score), np.std(increase_score)
        z_decrease_score = (decrease_score - mean_decrease_score) / std_decrease_score
        z_increase_score = (increase_score - mean_increase_score) / std_increase_score
        score = [ratio_of_decrease * z_decrease_score[i] + ratio_of_increase * z_increase_score[i] for i in range(len(z_decrease_score))]
    else:
        score = [ratio_of_decrease * decrease_score[i] + ratio_of_increase * increase_score[i] for i in range(len(decrease_score))]
    max_score = max(score)
    max_idx = score.index(max_score)

    plt.xticks(fontsize=tick_frontsize)
    plt.yticks(fontsize=tick_frontsize)

    t = list(range(len(decrease_score)))
    plt.plot(t, score, color, linewidth=LineWidth)
    plt.title(title, fontsize=title_frontsize)
    plt.xlabel("Inversion Step", fontsize=label_frontsize)
    plt.ylabel(y_label, fontsize=label_frontsize)
    plt.legend(legend_list, fontsize=legend_frontsize)
    plt.savefig(save_path)
    return max_idx + 1
