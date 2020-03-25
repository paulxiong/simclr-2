#!/bin/bash
python3 run.py --train_mode=pretrain --train_batch_size=512 \
--train_epochs=20 --learning_rate=1.0 --weight_decay=1e-6 \
--temperature=0.5 --data_dir=data  --dataset=mycervical \
 --image_size=100 --eval_split=test --resnet_depth=18 \
 --use_blur=False --color_jitter_strength=0.5 \
 --model_dir=./tmp/simclr_test --use_tpu=False
