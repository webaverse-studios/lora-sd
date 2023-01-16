import os
from PIL import Image
import shutil
# from google.colab import files
from tqdm import tqdm
from torch import autocast
import argparse

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
        help="Directory where LoRA trained model would be saved",
    )
    parser.add_argument(
        "--img_dir",
        type=str,
        # default="images",
        required=True,
        help="Directory where LoRA training images are present",
    )
    parser.add_argument(
        "--img_res",
        type=int,
        default=512,
        help="Resolution of resized images sent to training LoRA model",
    )




    parser.add_argument(
        "--train_steps",
        type=int,
        default=1000,
        help="Training steps for LoRA training",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=1,
        help="Batch size for LoRA training",
    )
    parser.add_argument(
        "--lr_unet",
        type=float,
        default=1e-4,
        help="Learning rate for LoRA UNET training",
    )
    parser.add_argument(
        "--lr_text",
        type=float,
        default=5e-5,
        help="Learning rate for LoRA Text Encoder training",
    )


    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()


    return args



def main(args):
    PRETRAINED_MODEL=args.pretrained_model_name_or_path #@param{type: 'string'}
    PROMPT=args.obj_prompt #@param{type: 'string'}

    OUTPUT_DIR=args.model_out_dir #@param{type: 'string'}
    IMAGES_FOLDER_OPTIONAL=args.img_dir #@param{type: 'string'}

    RESOLUTION= args.img_res #@param ["512", "576", "640", "704", "768", "832", "896", "960", "1024"]
    RESOLUTION=int(RESOLUTION)

    if PRETRAINED_MODEL == "":
        print('[1;31mYou should define the pretrained model.')

    else:
        if IMAGES_FOLDER_OPTIONAL=="" or len(os.listdir(IMAGES_FOLDER_OPTIONAL))==0:
            raise Exception('Training image folder not provided or empty')
        else:
            INSTANCE_DIR = IMAGES_FOLDER_OPTIONAL


    STEPS = args.train_steps #@param {type:"slider", min:0, max:10000, step:10}
    BATCH_SIZE = args.train_batch_size #@param {type:"slider", min:0, max:128, step:1}
    FP_16 = True #@param {type:"boolean"}


    LEARNING_RATE = args.lr_unet #@param {type:"number"}

    #DONE:Mandatory Text Encoder training !!!!
    TRAIN_TEXT_ENCODER = True #@param {type:"boolean"}

    LEARNING_RATE_TEXT_ENCODER = args.lr_text #@param {type:"number"}

    NEW_LEARNING_RATE = LEARNING_RATE / BATCH_SIZE
    NEW_LEARNING_RATE_TEXT_ENCODER = LEARNING_RATE_TEXT_ENCODER / BATCH_SIZE

    if FP_16:
        fp_16_arg = "fp16"
    else:
        fp_16_arg = "no"


    #Replacing images with resized PNGs of size 768x768 
    for filename in os.listdir(IMAGES_FOLDER_OPTIONAL):
        pil_img = Image.open(os.path.join(IMAGES_FOLDER_OPTIONAL, filename) )
        # print(type(pil_img))
        pil_img = pil_img.resize((RESOLUTION,RESOLUTION))

        filename_wo_ext = filename.split('.')[0]
        file_ext = filename.split('.')[-1]

        if file_ext=='png':
            continue
        else:
            pil_img.save( os.path.join(IMAGES_FOLDER_OPTIONAL, filename_wo_ext) + '.png', 'PNG' )
            os.remove(os.path.join(IMAGES_FOLDER_OPTIONAL, filename))

    os.system( f"accelerate launch './lora/training_scripts/train_lora_dreambooth.py' --pretrained_model_name_or_path={PRETRAINED_MODEL} --instance_data_dir={INSTANCE_DIR} --output_dir={OUTPUT_DIR} --instance_prompt={PROMPT} --train_text_encoder --use_8bit_adam --resolution={RESOLUTION} --mixed_precision={fp_16_arg} --train_batch_size={BATCH_SIZE} --gradient_accumulation_steps=1 --learning_rate={NEW_LEARNING_RATE} --learning_rate_text={NEW_LEARNING_RATE_TEXT_ENCODER} --color_jitter --lr_scheduler='constant' --lr_warmup_steps=0 --max_train_steps={STEPS}" )
    return

if __name__ == "__main__":
    args = parse_args()
    main(args)