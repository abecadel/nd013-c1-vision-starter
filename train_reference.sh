#!/bin/bash
#mkdir -p experiments/reference

#python edit_config.py --train_dir /waymo/splitted/train/ --eval_dir /waymo/splitted/val/ --batch_size 2 --checkpoint /app/experiments/pretrained_model/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/checkpoint/ --label_map /app/project/experiments/label_map.pbtxt

#mv pipeline_new.config experiments/reference/

python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
