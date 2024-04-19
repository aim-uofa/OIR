import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from sampler.ddim_scheduling import DDIMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel

# model config
# path to text CLIP 
clip_text_path = "/home/yangzhen/checkpoints/openai/clip-vit-base-patch16"
# path to Stable Diffusion
pretrained_model = '/home/yangzhen/checkpoints/huggingface/models/StableDiffusionModels/stable-diffusion-v1-4'

# ldm config
NUM_DDIM_STEPS = 50
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
MAX_NUM_WORDS = 77
GUIDANCE_SCALE = 7.5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained(pretrained_model, use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
ldm_stable.unet = UNet2DConditionModel.from_pretrained(pretrained_model, subfolder="unet").to(device)
ldm_stable.vae = AutoencoderKL.from_pretrained(pretrained_model, subfolder="vae").to(device)
tokenizer = ldm_stable.tokenizer
ldm_stable.enable_vae_tiling()
ldm_stable.enable_vae_slicing()



