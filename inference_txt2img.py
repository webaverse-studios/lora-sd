import os
from PIL import Image
import argparse
import torch
from lora_diffusion import monkeypatch_lora, tune_lora_scale
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline 


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
        default="0.6",
        help="LORA_SCALE_UNET",
    )
    parser.add_argument(
        "--lora_scale_text",
        type=float,
        default="0.8",
        help="LORA_SCALE_TEXT_ENCODER",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        default="7.6",
        help="GUIDANCE",
    )
    parser.add_argument(
        "--num_steps",
        type=int,
        default="50",
        help="GUIDANCE",
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

def main(args):

    pipe = StableDiffusionPipeline.from_pretrained(args.pretrained_model_name_or_path, torch_dtype=torch.float16, safety_checker = None).to("cuda")
    # pipe = StableDiffusionImg2ImgPipeline.from_pretrained(PRETRAINED_MODEL, torch_dtype=torch.float16, safety_checker = None).to("cuda")
    monkeypatch_lora(pipe.unet, torch.load(os.path.join(args.model_out_dir, "lora_weight.pt")))
    monkeypatch_lora(pipe.text_encoder, torch.load(os.path.join(args.model_out_dir, "lora_weight.text_encoder.pt")), target_replace_module=["CLIPAttention"])

    print(f'Loaded model with trigger word \'{args.obj_prompt}\'')

    INFERENCE_PROMPT = args.prompt
    INFERENCE_PROMPT = [x.strip() for x in INFERENCE_PROMPT.split(':')]

    LORA_SCALE_UNET = args.lora_scale_unet 
    LORA_SCALE_TEXT_ENCODER = args.lora_scale_text
    GUIDANCE = args.cfg_scale
    NUM_INFERENCE_STEPS = args.num_steps

    tune_lora_scale(pipe.unet, LORA_SCALE_UNET)
    tune_lora_scale(pipe.text_encoder, LORA_SCALE_TEXT_ENCODER)

    images = pipe(INFERENCE_PROMPT, num_inference_steps=NUM_INFERENCE_STEPS, guidance_scale=GUIDANCE).images

    # if not os.path.exists(args.out_dir):
    #     os.makedirs(args.out_dir)

    # for idx,img in enumerate(images):
    #     img.save(os.path.join(args.out_dir, f'lora_{args.obj_prompt}_{idx}.png'))

    return images


if __name__ == "__main__":
    args = parse_args()
    main(args)