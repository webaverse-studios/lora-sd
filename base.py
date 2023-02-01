import os 
# import lora_train
from lora_diffusion import cli_lora_pti
import inference_txt2img
import inference_img2img


def train_lora(MODEL_NAME = "stable-diffusion-2-1-base",\
    INSTANCE_DATA_DIR = 'images',\
    OUTPUT_DIR = 'lora_wassy_model',\
    RESOLUTION = 512,\
    KEYWORD = "wassy", \
    INIT_TOKEN = None,
    LR_SCHEDULER = 'linear',
    UNET_LR = 1e-4,
    TEXT_ENC_LR = 1e-5,
    TI_LR = 5e-4,
    CONT_INV_LR = 1e-4,
    TI_TUNING_STEPS = 1000,
    ):

    TRAIN_BATCH_SIZE = 1

    if INIT_TOKEN is not None and len(INIT_TOKEN.strip().split())==1:
        cli_lora_pti.train(
        pretrained_model_name_or_path=MODEL_NAME ,
        instance_data_dir=INSTANCE_DATA_DIR ,
        output_dir=OUTPUT_DIR ,
        placeholder_tokens=KEYWORD , 
        placeholder_token_at_data= f"TOKEN|{KEYWORD}",
        initializer_tokens=INIT_TOKEN,
        train_text_encoder = True ,
        resolution=RESOLUTION ,
        train_batch_size=TRAIN_BATCH_SIZE ,
        gradient_accumulation_steps=4 ,
        gradient_checkpointing = True ,
        scale_lr = True ,
        learning_rate_unet=UNET_LR ,
        learning_rate_text=TEXT_ENC_LR ,
        learning_rate_ti=TI_LR ,
        color_jitter = True,
        lr_scheduler=LR_SCHEDULER ,
        lr_warmup_steps=0 ,
        lr_scheduler_lora=LR_SCHEDULER ,
        lr_warmup_steps_lora=100 ,
        save_steps=100 ,
        max_train_steps_ti=TI_TUNING_STEPS ,
        max_train_steps_tuning=TI_TUNING_STEPS ,
        perform_inversion=True ,
        clip_ti_decay = True ,
        weight_decay_ti=0.000 ,
        weight_decay_lora=0.001 ,
        continue_inversion = True ,
        continue_inversion_lr=CONT_INV_LR ,
        device="cuda:0" ,
        lora_rank=1
        )
    else:
        cli_lora_pti.train(
        pretrained_model_name_or_path=MODEL_NAME ,
        instance_data_dir=INSTANCE_DATA_DIR ,
        output_dir=OUTPUT_DIR ,
        placeholder_tokens=KEYWORD , 
        placeholder_token_at_data= f"TOKEN|{KEYWORD}",
        train_text_encoder = True ,
        resolution=RESOLUTION ,
        train_batch_size=TRAIN_BATCH_SIZE ,
        gradient_accumulation_steps=4 ,
        gradient_checkpointing = True ,
        scale_lr = True ,
        learning_rate_unet=UNET_LR ,
        learning_rate_text=TEXT_ENC_LR ,
        learning_rate_ti=TI_LR ,
        color_jitter = True,
        lr_scheduler=LR_SCHEDULER ,
        lr_warmup_steps=0 ,
        lr_scheduler_lora=LR_SCHEDULER ,
        lr_warmup_steps_lora=100 ,
        save_steps=100 ,
        max_train_steps_ti=TI_TUNING_STEPS ,
        max_train_steps_tuning=TI_TUNING_STEPS ,
        perform_inversion=True ,
        clip_ti_decay = True ,
        weight_decay_ti=0.000 ,
        weight_decay_lora=0.001 ,
        continue_inversion = True ,
        continue_inversion_lr=CONT_INV_LR ,
        device="cuda:0" ,
        lora_rank=1
        )

    # lora_train_args = train_lora_dreambooth.parse_args(input_args=['--pretrained_model_name_or_path', str(MODEL_NAME),\
    #     '--instance_data_dir', str(INSTANCE_DATA_DIR), '--output_dir', str(OUTPUT_DIR), '--instance_prompt', str(KEYWORD), '--train_text_encoder',\
    #     '--resolution', str(RESOLUTION), '--mixed_precision', str(fp_16_arg), '--train_batch_size', str(TRAIN_BATCH_SIZE),\
    #     '--gradient_accumulation_steps', str(GRAD_ACC_STEPS), '--learning_rate', str(UNET_LR), '--learning_rate_text', str(TEXT_ENC_LR), \
    #     '--lr_scheduler', str(LR_SCHEDULER), '--lr_warmup_steps', str(0), '--max_train_steps', str(TRAIN_STEPS), '--color_jitter', '--use_8bit_adam'])

    # train_lora_dreambooth.main(lora_train_args)


    # os.system( f"python {'lora_train.py'} --pretrained_model_name_or_path={MODEL_NAME} --instance_data_dir={INSTANCE_DATA_DIR}\
    # --output_dir={OUTPUT_DIR} --instance_prompt='{KEYWORD}' --train_text_encoder --use_8bit_adam --resolution={RESOLUTION} --mixed_precision={fp_16_arg} --train_batch_size={TRAIN_BATCH_SIZE} --gradient_accumulation_steps={GRAD_ACC_STEPS} \
    # --learning_rate={UNET_LR} --learning_rate_text={TEXT_ENC_LR} --color_jitter --lr_scheduler='constant' --lr_warmup_steps=0 --max_train_steps={TRAIN_STEPS}")
    
    return

def inference_lora_txt2img(MODEL_NAME = "stable-diffusion-2-1-base",\
    KEYWORD = "wassy", \
    OUTPUT_DIR = 'lora_wassy_model',\
    PROMPT = "wassy in style of Van Gogh: wassy in style of Monet ",
    TI_TUNING_STEPS = 1000,
    ):

    lora_txt2img_args = inference_txt2img.parse_args(input_args=['--pretrained_model_name_or_path', MODEL_NAME,\
        '--obj_prompt', KEYWORD, '--model_out_dir', OUTPUT_DIR, '--prompt', PROMPT, '--ti_tuning_steps', TI_TUNING_STEPS])

    images = inference_txt2img.main(lora_txt2img_args)
    return images

    # os.system( f"python {'inference.py'} --pretrained_model_name_or_path={MODEL_NAME} --obj_prompt={KEYWORD} --model_out_dir={OUTPUT_DIR} --prompt={PROMPT}")
    return

def inference_lora_img2img(MODEL_NAME = "stable-diffusion-2-1-base",\
    KEYWORD = "wassy", \
    OUTPUT_DIR = 'lora_wassy_model',\
    PROMPT = "wassy in style of Van Gogh: wassy in style of Monet ",\
    INIT_IMG = 'local_img.png',
    TI_TUNING_STEPS = 1000,
    ):

    lora_img2img_args = inference_img2img.parse_args(input_args=['--pretrained_model_name_or_path', MODEL_NAME,\
        '--obj_prompt', KEYWORD, '--model_out_dir', OUTPUT_DIR, '--prompt', PROMPT, '--init_image', INIT_IMG, '--ti_tuning_steps', TI_TUNING_STEPS])

    images = inference_img2img.main(lora_img2img_args)
    return images