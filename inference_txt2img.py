import os
from PIL import Image
import argparse
import torch
from lora_diffusion import patch_pipe, tune_lora_scale
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline 

pipe = None
args = None

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        # default="stabilityai/stable-diffusion-2-1-base",
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--obj_prompt",
        type=str,
        # default="sksobjkt",
        required=True,
        help="Dreambooth trigger word",
    )
    parser.add_argument(
        "--model_out_dir",
        type=str,
        # default="lora_model",
        required=True,
        help="Directory where LoRA trained model is saved",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        # default="photo:of:a",
        required=True,
        help="Prompts passed to the LoRA model for Text to image generation, should be separated by \':\'.",
    )


    parser.add_argument(
        "--img_res",
        type=int,
        default=768,
        help="Resolution of resized images sent to training/img2img inferring LoRA model",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="lora_img_outs",
        help="Directory where LoRA images would be saved",
    )


    parser.add_argument(
        "--lora_scale_unet",
        type=float,
        default=1.0,
        help="LORA_SCALE_UNET",
    )
    parser.add_argument(
        "--lora_scale_text",
        type=float,
        default=1.0,
        help="LORA_SCALE_TEXT_ENCODER",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default=7.6,
        help="GUIDANCE",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default="50",
        help="GUIDANCE",
    )
    parser.add_argument(
        '--ti_tuning_steps',
        type=int,
        default=1000,
        help="number of steps each that model was trained on inversion and ti"
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args 

def load_image(image_path, size = 768):
  init_img = Image.open(image_path).convert("RGB").resize((size, size))
  #returns a PIL Image
  return init_img

def cbk(step, timestep, latents):
    val = timestep.item() / 1000
    it = 1 - val  # it starts from 0 to 1
    tune_unet = 0 if it < 0.3 else LORA_SCALE_UNET 
    tune_text = 0 if it < 0.3 else LORA_SCALE_TEXT

    tune_lora_scale(pipe.unet, tune_unet)
    tune_lora_scale(pipe.text_encoder, tune_text)

def main(args):
    global pipe 

    global LORA_SCALE_UNET 
    LORA_SCALE_UNET = args.lora_scale_unet
    global LORA_SCALE_TEXT
    LORA_SCALE_TEXT  = args.lora_scale_text

    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker = None).to("cuda")
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(PRETRAINED_MODEL, torch_dtype=torch.float16, safety_checker = None).to("cuda")

    patch_pipe(
        pipe,
        os.path.join(args.model_out_dir, f"step_{args.ti_tuning_steps}.safetensors"),
        patch_text=True,
        patch_ti=True,
        patch_unet=True,
    )

    print(f'Loaded model with trigger word \'{args.obj_prompt}\'')

    INFERENCE_PROMPT = args.prompt
    INFERENCE_PROMPT = [x.strip() for x in INFERENCE_PROMPT.split(':')]

    GUIDANCE = args.cfg_scale
    NUM_INFERENCE_STEPS = args.num_steps

    tune_lora_scale(pipe.unet, 1.00)
    tune_lora_scale(pipe.text_encoder, 1.00)

    images = pipe(INFERENCE_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, callback=cbk, callback_steps=1, guidance_scale=GUIDANCE).images

    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)

    # for idx,img in enumerate(images):
    #     img.save(os.path.join(args.out_dir, f'lora_{args.obj_prompt}_{idx}.png'))

    return images


if __name__ == "__main__":
    args = parse_args()
    main(args)