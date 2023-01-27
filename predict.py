from cog import BasePredictor, BaseModel, File, Input, Path
from base import train_lora, inference_lora_txt2img, inference_lora_img2img
from PIL import Image

import base64

import urllib.request

from typing import Any, List
import torch 

print('cuda status is',torch.cuda.is_available())

class Output(BaseModel):
    file: File
    ip: str


class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""
        print('Stable Diffusion started!')

    def predict(
        self,
        mode: Path = Input(description="Mode is assumed to be either \'train\' or \'inference\'", default = 'train' ),
        model_path: Path = Input(description="Path to the pretrained base Stable Diffusion model directory", default = './stable-diffusion-2-1-base' ),
        model_out_dir: Path = Input(description="Path to the directory containing LoRA checkpoints/ where LoRA checkpoints should be saved ", default = './lora_sksperson_model' ),
        keyword: Path = Input(description="The word embedding to be added by the LoRA using the reference images in `instance_data_dir`", default = 'sksperson' ),
        instance_data_dir: str = Input(description="Directory containing the sample images for training, valid only for `train` mode", default='./images'),
        resolution: int = Input(description="Resolution to resize the scripts to, valid only for `train` mode", default=512),
        unet_lr: float = Input(description="LR for training UNet, valid only for `train` mode", default=1e-4),
        text_encoder_lr: float = Input(description="LR for training Textencoder, valid only for `train` mode", default=5e-5),
        num_train_steps: int = Input(description="Number of training steps, usually = num_images * 300, valid only for `train` mode", default=2000),
        inference_prompt: str = Input(description="Prompt for running inference with the LoRA model, valid only for `inference` mode", default='a photo of an sksperson'),
        init_img: str = Input(description="Init image for running inference with the LoRA model with img2img, valid only for `inference` mode", default=None),
    ) -> Any:
        """Run a single prediction on the model"""
        try:
            if mode.lower().strip() is 'train':
                train_lora(MODEL_NAME = model_path,\
                INSTANCE_DATA_DIR = instance_data_dir,\
                OUTPUT_DIR = model_out_dir,\
                RESOLUTION = resolution,\
                KEYWORD = keyword, \
                UNET_LR = unet_lr,\
                TEXT_ENC_LR = text_encoder_lr,\
                TRAIN_STEPS = num_train_steps
                )

            else:
                if init_img is None:
                    inference_lora_txt2img(MODEL_NAME = model_path,\
                    KEYWORD = keyword, \
                    OUTPUT_DIR = model_out_dir,\
                    PROMPT = inference_prompt
                    )
                else:
                    inference_lora_txt2img(MODEL_NAME = model_path,\
                    KEYWORD = keyword, \
                    OUTPUT_DIR = model_out_dir,\
                    PROMPT = inference_prompt
                    )
        except Exception as e:
            return f"Error: {e}"