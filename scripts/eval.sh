#!/bin/bash
# eavl.sh GPU_ID
GPU_ID=$1
python eval.py\
        --batch-size 128\
        --scale 448 448\
        --gpu $GPU_ID\
        --dataset iwildcam\
        --root_dir data/iwildcam\
        --eval_split id_test test\
        --metric acc loss macro_f1\
        --y_reorder original\
        --model_path    \
            \
        --draw_label_probability\
        