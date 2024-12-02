import os
import re
import time
import subprocess

cmd = """
export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
GPU_COUNT=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=`seq -s, 0 $((GPU_COUNT-1))`
torchrun --nproc_per_node=$GPU_COUNT\
            main.py\
            --result-dir results/lambda_search/{configname}/\
            --arch resnet50\
            --seed {seed}\
            --method PU\
            --weights {weights}\
            --epochs $epochs\
            --batch-size 48\
            --scale 224 224\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss CrossEntropyLoss\
            --optimizer AdamW\
            --lr 5e-5\
            --weight-decay 1e-8\
            --scheduler CosineAnnealingLR\
            --scheduler_kwargs T_max=$epochs\
            --dataset terrainc\
            --root_dir data/terrainc\
            --train_split train\
            --eval_split val test\
            --model_selection_split val\
            --model_selection_metric macro_f1\
            --metric acc loss macro_f1\
            --distributed
"""

for root, dirs, files in os.walk("weights/terrainc"):
    for filename in files:
        if filename.endswith(".npy"):
            for seed in range(1, 3):
                command = cmd.format(
                    weights=os.path.join(root, filename), configname=filename, seed=seed
                )
                os.system(command)
