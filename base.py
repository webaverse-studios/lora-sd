import os 
import lora_train
import inference_txt2img
import inference_img2img

def train_lora(MODEL_NAME = "stable-diffusion-2-1-base",\
    INSTANCE_DATA_DIR = 'images',\
    OUTPUT_DIR = 'lora_wassy_model',\
    RESOLUTION = 512,\
    KEYWORD = "wassy", \
    UNET_LR = 1e-4,
    TEXT_ENC_LR = 5e-5,
    TRAIN_STEPS = 1200
    ):

    fp_16_arg = "fp16"
    TRAIN_BATCH_SIZE = 1
    GRAD_ACC_STEPS = 1

    # lora_train_args = lora_train.parse_args(input_args=['--pretrained_model_name_or_path', MODEL_NAME,\
    #     '--instance_data_dir', INSTANCE_DATA_DIR, '--output_dir', OUTPUT_DIR, '--instance_prompt', KEYWORD, '--train_text_encoder',\
    #     '--use_8bit_adam', '--resolution', RESOLUTION, '--mixed_precision', fp_16_arg, '--train_batch_size', TRAIN_BATCH_SIZE,\
    #     '--gradient_accumulation_steps', GRAD_ACC_STEPS, '--learning_rate', UNET_LR, '--learning_rate_text', TEXT_ENC_LR, \
    #     '--color_jitter', '--lr_scheduler', 'constant', '--lr_warmup_steps', 0, '--max_train_steps', TRAIN_STEPS])

    # lora_train.main(lora_train_args)

    os.system( f"python {'lora_train.py'} --pretrained_model_name_or_path={MODEL_NAME} --instance_data_dir={INSTANCE_DATA_DIR}\
    --output_dir={OUTPUT_DIR} --instance_prompt='{KEYWORD}' --train_text_encoder --use_8bit_adam --resolution={RESOLUTION} --mixed_precision={fp_16_arg} --train_batch_size={TRAIN_BATCH_SIZE} --gradient_accumulation_steps={GRAD_ACC_STEPS} \
    --learning_rate={UNET_LR} --learning_rate_text={TEXT_ENC_LR} --color_jitter --lr_scheduler='constant' --lr_warmup_steps=0 --max_train_steps={TRAIN_STEPS}")
    
    return

def inference_lora_txt2img(MODEL_NAME = "stable-diffusion-2-1-base",\
    KEYWORD = "wassy", \
    OUTPUT_DIR = 'lora_wassy_model',\
    PROMPT = "wassy in style of Van Gogh: wassy in style of Monet "
    ):

    lora_txt2img_args = inference_txt2img.parse_args(input_args=['--pretrained_model_name_or_path', MODEL_NAME,\
        '--obj_prompt', KEYWORD, '--model_out_dir', OUTPUT_DIR, '--prompt', PROMPT])

    images = inference_txt2img.main(lora_txt2img_args)
    return images

    # os.system( f"python {'inference.py'} --pretrained_model_name_or_path={MODEL_NAME} --obj_prompt={KEYWORD} --model_out_dir={OUTPUT_DIR} --prompt={PROMPT}")
    return

def inference_lora_img2img(MODEL_NAME = "stable-diffusion-2-1-base",\
    KEYWORD = "wassy", \
    OUTPUT_DIR = 'lora_wassy_model',\
    PROMPT = "wassy in style of Van Gogh: wassy in style of Monet ",\
    INIT_IMG = 'local_img.png'
    ):

    lora_img2img_args = inference_txt2img.parse_args(input_args=['--pretrained_model_name_or_path', MODEL_NAME,\
        '--obj_prompt', KEYWORD, '--model_out_dir', OUTPUT_DIR, '--prompt', PROMPT, '--init_image', INIT_IMG])

    images = inference_txt2img.main(lora_img2img_args)
    return images