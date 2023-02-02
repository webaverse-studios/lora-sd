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
        model_path: Path = Input(description="Path to the pretrained base Stable Diffusion model directory", default = './stable-diffusion-v1-4' ),
        model_out_dir: Path = Input(description="Path to the directory containing LoRA checkpoints/ where LoRA checkpoints should be saved ", default = './lora_sksperson_model' ),
        keyword: Path = Input(description="The word embedding to be added by the LoRA using the reference images in `instance_data_dir`", default = 'sksperson' ),
        instance_data_dir: str = Input(description="Directory containing the sample images for training, valid only for `train` mode", default='./images'),
        resolution: int = Input(description="Resolution to resize the scripts to, valid only for `train` mode", default=512),
        lr_scheduler: str = Input(description="lr scheduler name, valid only for `train` mode", default='linear'),
        init_token: str = Input(description="init token for LoRA training, valid only for `train` mode", default=None),
        unet_lr: float = Input(description="LR for training UNet, valid only for `train` mode", default=9e-4),
        ti_lr: float = Input(description="LR for training UNet, valid only for `train` mode", default=5e-4),
        cont_lr: float = Input(description="LR for training UNet, valid only for `train` mode", default=1e-4),
        text_encoder_lr: float = Input(description="LR for training Textencoder, valid only for `train` mode", default=1e-5),
        ti_tuning_steps: int = Input(description="Number of training steps for both inversion and tuning, valid only for `train` mode", default=1000),
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
                LR_SCHEDULER = lr_scheduler, \
                INIT_TOKEN = init_token, \
                UNET_LR = unet_lr,\
                TEXT_ENC_LR = text_encoder_lr, \
                TI_LR = ti_lr,
                CONT_INV_LR = cont_lr,
                TI_TUNING_STEPS = ti_tuning_steps,
                )

            else:
                if init_img is None:
                    inference_lora_txt2img(MODEL_NAME = model_path,\
                    KEYWORD = keyword, \
                    OUTPUT_DIR = model_out_dir,\
                    PROMPT = inference_prompt, \
                    TI_TUNING_STEPS = ti_tuning_steps,
                    )
                else:
                    inference_lora_img2img(MODEL_NAME = model_path,\
                    KEYWORD = keyword, \
                    OUTPUT_DIR = model_out_dir,\
                    PROMPT = inference_prompt,\
                    INIT_IMG = init_img,
                    TI_TUNING_STEPS = ti_tuning_steps,
                    )
        except Exception as e:
            return f"Error: {e}"

            