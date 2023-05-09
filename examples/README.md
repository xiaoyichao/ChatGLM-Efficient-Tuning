# fine-tuning
CUDA_VISIBLE_DEVICES=0 python src/finetune.py \
    --do_train \
    --dataset alpaca_gpt4_zh \
    --finetuning_type lora \
    --output_dir path_to_checkpoint \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_total_limit 2 \
    --save_strategy epoch \
    --learning_rate 5e-5 \
    --num_train_epochs 1.0 \
    --fp16

nohup bash finetune.sh >/dev/null 2>&1 &
