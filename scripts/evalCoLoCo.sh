#!/bin/bash
# eavl.sh GPU_ID
GPU_ID=2
python eval.py\
        --arch resnet50\
        --method DU\
        --batch-size 32\
        --scale 448 448\
        --gpu $GPU_ID\
        --dataset iwildcam\
        --root_dir data/iwildcam\
        --eval_split id_test test\
        --metric acc loss macro_f1\
        --model_path results/report/bw/DU/ptw_ablation/ptw_0.01
        # --model_path results/report/bw/DU/ptw_ablation\
        # --draw_label_probability\