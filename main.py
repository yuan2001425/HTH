import os

# os.environ["NPROC_PER_NODE"] = "4"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

os.environ["NPROC_PER_NODE"] = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"

# command = "swift sft " \
#           "--model_type deepseek-vl-7b-chat " \
#           "--dataset /media/oem/12T/HZY/HTH/dataset.json " \
#           "--sft_type full " \
#           "--lora_target_modules ALL " \
#           "--use_flash_attn true " \
#           "--deepspeed default-zero2 " \
#           "--model_name DEMO_HTH_MULTI " \
#           "--model_author X.F.Liang " \
#           "--output_dir /media/oem/12T/HZY/HTH/ " \
#           "--num_train_epochs 20"


command = "swift sft " \
          "--model_type deepseek-vl-7b-chat " \
          "--dataset /media/oem/12T/HZY/HTH/dataset3.json " \
          "--model_name DEMO_HTH_MULTI " \
          "--model_author X.F.Liang " \
          "--output_dir /media/oem/12T/HZY/HTH/ " \
          "--num_train_epochs 20"

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#
# command = "swift infer " \
#           "--ckpt_dir deepseek-vl-7b-chat/v1-20240518-145315/checkpoint-380 " \
#           "--load_dataset_config false "

os.system(command)
