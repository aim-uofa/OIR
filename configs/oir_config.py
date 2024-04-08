import sys
sys.path.append("/home/yangzhen/code/DynamicInversion")
import torch
from diffusers import StableDiffusionPipeline
from diffusers import DDIMScheduler
from sampler.ddim_scheduling import DDIMScheduler
from models.autoencoder_kl import AutoencoderKL
from models.unet_2d_condition import UNet2DConditionModel
from configs.user_input import args


# ldm config
NUM_DDIM_STEPS = 2
scheduler = DDIMScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", clip_sample=False, set_alpha_to_one=False)
MY_TOKEN = ''
LOW_RESOURCE = False 
MAX_NUM_WORDS = 77
GUIDANCE_SCALE = 7.5
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
ldm_stable = StableDiffusionPipeline.from_pretrained(args['pretrained_model'], use_auth_token=MY_TOKEN, scheduler=scheduler).to(device)
unet = UNet2DConditionModel.from_pretrained(args['pretrained_model'], subfolder="unet").to(device)
vae = AutoencoderKL.from_pretrained(args['pretrained_model'], subfolder="vae").to(device)
ldm_stable.unet = unet
ldm_stable.vae = vae
tokenizer = ldm_stable.tokenizer
ldm_stable.enable_vae_tiling()
ldm_stable.enable_vae_slicing()

# oir config
mask_at_where = 'latent' # pixel/latent
unmask_area_recover_method = 'inversion_latent' # inversion_latent/guided_latent
normalize_method = 'max-min' # z-score
background_mask_method = 'segmentation' # segmentation/detection
background_box_type = 'global_box' # local_box/global_box
object_mask_method = 'segmentation' # segmentation/detection
object_box_type = 'global_box' # local_box/global_box
blended_all_step_after_background_sweet_point = False
use_negative_prompt_inversion=True
use_reinversion = True
# reassembly_step从1开始，不能为0，就是background_sweet_point
reassembly_step = 10 # 30 20 10 5 

