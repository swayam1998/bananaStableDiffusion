# In this file, we define download_model
# It runs during container build time to get model weights built into the container

from diffusers import StableDiffusionPipeline, LMSDiscreteScheduler, AutoencoderKL
from diffusers import UNet2DConditionModel, PNDMScheduler, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
import os

def download_model():
    # do a dry run of loading the huggingface model, which will download weights at build time
    #Set auth token which is required to download stable diffusion model weights
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=HF_AUTH_TOKEN)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=HF_AUTH_TOKEN)

    # 4. Create a scheduler for inference
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000
        )

if __name__ == "__main__":
    download_model()
