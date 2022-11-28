import torch
from torch import autocast
# from diffusers import StableDiffusionPipeline, PNDMScheduler
from diffusers import UNet2DConditionModel,LMSDiscreteScheduler, AutoencoderKL, LMSDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from tqdm.auto import tqdm
from PIL import Image

import base64
from io import BytesIO
import os

device = 'cuda'

# Init is ran on server startup
# Load your model to GPU as a global variable here using the variable name "model"
def init():
    global vae, tokenizer, text_encoder, unet, scheduler
    HF_AUTH_TOKEN = os.getenv("HF_AUTH_TOKEN")

    # 1. Load the autoencoder model which will be used to decode the latents into image space. 
    vae = AutoencoderKL.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='vae', use_auth_token=HF_AUTH_TOKEN)
    vae = vae.to(device)

    # 2. Load the tokenizer and text encoder to tokenize and encode the text. 
    tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14')
    text_encoder = text_encoder.to(device)

    # 3. The UNet model for generating the latents.
    unet = UNet2DConditionModel.from_pretrained(
        'CompVis/stable-diffusion-v1-4', subfolder='unet', use_auth_token=HF_AUTH_TOKEN)
    unet = unet.to(device)

    # 4. Create a scheduler for inference
    scheduler = LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012,
        beta_schedule='scaled_linear', num_train_timesteps=1000)
    
    # this will substitute the default PNDM scheduler for K-LMS  
    # lms = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear")

    # model = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", scheduler=lms, use_auth_token=HF_AUTH_TOKEN).to("cuda")

def get_text_embeds(prompt):
    global tokenizer, text_encoder
    # Tokenize text and get embeddings
    text_input = tokenizer(
        prompt, padding='max_length', max_length=tokenizer.model_max_length,
        truncation=True, return_tensors='pt')
    with torch.no_grad():
        text_embeddings = text_encoder(text_input.input_ids.to(device))[0]

    # Do the same for unconditional embeddings
    uncond_input = tokenizer(
        [''] * len(prompt), padding='max_length',
        max_length=tokenizer.model_max_length, return_tensors='pt')
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]

    # Cat for final embeddings
    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
    return text_embeddings

def produce_latents(text_embeddings, height=512, width=512, num_inference_steps=50, guidance_scale=7.5, latents=None):
    global scheduler, unet

    if latents is None:
        latents = torch.randn((text_embeddings.shape[0] // 2, unet.in_channels, \
                            height // 8, width // 8))
    latents = latents.to(device)

    scheduler.set_timesteps(num_inference_steps)
    latents = latents * scheduler.sigmas[0]

    with autocast('cuda'):
        for i, t in tqdm(enumerate(scheduler.timesteps)):
            # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
            latent_model_input = torch.cat([latents] * 2)
            sigma = scheduler.sigmas[i]
            latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

            # predict the noise residual
            with torch.no_grad():
                noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings)['sample']

            # perform guidance
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, i, latents)['prev_sample']
    
    return latents

def decode_img_latents(latents):
    global vae

    latents = 1 / 0.18215 * latents

    with torch.no_grad():
        imgs = vae.decode(latents)

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    imgs = imgs.detach().cpu().permute(0, 2, 3, 1).numpy()
    imgs = (imgs * 255).round().astype('uint8')
    pil_images = [Image.fromarray(image) for image in imgs]
    return pil_images

def prompt_to_img(prompts, height=512, width=512, num_inference_steps=50,guidance_scale=7.5, latents=None):
  if isinstance(prompts, str):
    prompts = [prompts]

  # Prompts -> text embeds
  text_embeds = get_text_embeds(prompts)

  # Text embeds -> img latents
  latents = produce_latents(
      text_embeds, height=height, width=width, latents=latents,
      num_inference_steps=num_inference_steps, guidance_scale=guidance_scale)
  
  # Img latents -> imgs
  imgs = decode_img_latents(latents)

  return imgs

# Inference is ran for every server call
# Reference your preloaded global model variable here.
def inference(model_inputs:dict) -> dict:
    # global model

    # Parse out your arguments
    prompt = model_inputs.get('prompt', None)
    height = model_inputs.get('height', 512)
    width = model_inputs.get('width', 512)
    num_inference_steps = model_inputs.get('num_inference_steps', 50)
    guidance_scale = model_inputs.get('guidance_scale', 7.5)
    input_seed = model_inputs.get("seed",None)
    
    #If "seed" is not sent, we won't specify a seed in the call
    generator = None
    if input_seed != None:
        generator = torch.Generator("cuda").manual_seed(input_seed)
    
    if prompt == None:
        return {'message': "No prompt provided"}

    #Run the pipeline
    image = prompt_to_img(prompt, height, width, num_inference_steps, guidance_scale)[0]
    
    # Run the model
    # with autocast("cuda"):
    #     image = model(prompt,height=height,width=width,num_inference_steps=num_inference_steps,guidance_scale=guidance_scale,generator=generator)["sample"][0]
    
    buffered = BytesIO()
    image.save(buffered,format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Return the results as a dictionary
    return {'image_base64': image_base64}
