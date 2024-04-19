import numpy as np
from tqdm import trange
import torch, cv2, os, time
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
from IPython.display import display
from PIL import Image, ImageDraw, ImageFont
from typing import Optional, Union, Tuple, List, Callable, Dict

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img

def view_images(images, save_path, file_name, num_rows=1, offset_ratio=0.02):
    if type(images) is list:
        num_empty = len(images) % num_rows
    elif images.ndim == 4:
        num_empty = images.shape[0] % num_rows
    else:
        images = [images]
        num_empty = 0
    empty_images = np.ones(images[0].shape, dtype=np.uint8) * 255
    images = [image.astype(np.uint8) for image in images] + [empty_images] * num_empty
    num_items = len(images)
    h, w, c = images[0].shape
    offset = int(h * offset_ratio)
    num_cols = num_items // num_rows
    image_ = np.ones((h * num_rows + offset * (num_rows - 1),
                      w * num_cols + offset * (num_cols - 1), 3), dtype=np.uint8) * 255
    for i in range(num_rows):
        for j in range(num_cols):
            image_[i * (h + offset): i * (h + offset) + h:, j * (w + offset): j * (w + offset) + w] = images[
                i * num_cols + j]
    pil_img = Image.fromarray(image_)
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    pil_img.save(save_path + file_name)

def diffusion_step(model, latents, context, t, guidance_scale, low_resource=False):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(
            latents_input, 
            t, 
            encoder_hidden_states=context,
        )["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents)["prev_sample"]
    return latents

def diffusion_step_parallel(model, latents, context, t, guidance_scale, low_resource=False, use_parallel=True):
    if low_resource:
        noise_pred_uncond = model.unet(latents, t, encoder_hidden_states=context[0])["sample"]
        noise_prediction_text = model.unet(latents, t, encoder_hidden_states=context[1])["sample"]
    else:
        latents_input = torch.cat([latents] * 2)
        noise_pred = model.unet(
            latents_input, 
            torch.cat([t, t]), 
            encoder_hidden_states=context,
            use_parallel=use_parallel,
        )["sample"]
        noise_pred_uncond, noise_prediction_text = noise_pred.chunk(2)

    noise_pred = noise_pred_uncond + guidance_scale * (noise_prediction_text - noise_pred_uncond)
    latents = model.scheduler.step(noise_pred, t, latents, use_parallel=True)["prev_sample"]
    return latents

def latent2image(vae, latents):
    latents = 1 / 0.18215 * latents
    image = vae.decode(latents)['sample']
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.cpu().permute(0, 2, 3, 1).detach().numpy()
    image = (image * 255).astype(np.uint8)
    return image

def image2latent(vae, image):
    with torch.no_grad():
        if type(image) is Image:
            image = np.array(image)
        if type(image) is torch.Tensor and image.dim() == 4:
            latents = image
        else:
            image = torch.from_numpy(image).float() / 127.5 - 1
            image = image.permute(2, 0, 1).unsqueeze(0).to(vae.device)
            latents = vae.encode(image)['latent_dist'].mean
            latents = latents * 0.18215
    return latents

def init_latent(latent, model, height, width, batch_size):
    if latent is None:
        latent = torch.randn((1, model.unet.in_channels, height // 8, width // 8))
    latents = latent.expand(batch_size, model.unet.in_channels, height // 8, width // 8).to(model.device)
    return latent, latents

def init_latent_parallel(latent, model, height, width, batch_size, all_latents, num_ddim_steps):
    latents = all_latents[1]
    for i in range(1, num_ddim_steps // 2):
        latents = torch.cat([latents, all_latents[i + 1]])
    return latent, latents

def load_512(image_path, left=0, right=0, top=0, bottom=0):
    if type(image_path) is str:
        image = np.array(Image.open(image_path))[:, :, :3]
    else:
        image = image_path
    h, w, c = image.shape
    left = min(left, w - 1)
    right = min(right, w - left - 1)
    top = min(top, h - left - 1)
    bottom = min(bottom, h - top - 1)
    image = image[top:h - bottom, left:w - right]
    h, w, c = image.shape
    if h < w:
        offset = (w - h) // 2
        image = image[:, offset:offset + h]
    elif w < h:
        offset = (h - w) // 2
        image = image[offset:offset + w]
    image = np.array(Image.fromarray(image).resize((512, 512)))
    return image

def change_images_to_file(
        generated_images_path,
        image_name,
        num_steps,
):
    images = Image.open(os.path.join(generated_images_path, image_name))
    images = np.array(images)
    for i in trange(1, num_steps + 1):
        fig_name = str(i).zfill(4) + '.png'
        splice_image_path = os.path.join(generated_images_path, fig_name)
        left = 522 * (i - 1)
        right = left + 512
        image = images[:, left: right, :]
        image = Image.fromarray(image)
        image.save(splice_image_path)
