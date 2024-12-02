import os
import re
import time
import subprocess

from analysis.avg_std import selection

cmd = """
export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
GPU_COUNT=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=`seq -s, 0 $((GPU_COUNT-1)) | paste -sd ''`
torchrun --nproc_per_node=$GPU_COUNT\
            main.py\
            --method PU\
            --arch resnet50\
            --weights {weights}\
            --seed {seed}\
            --epochs $epochs\
            --batch-size 48\
            --scale 448 448\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss CrossEntropyLoss\
            --optimizer AdamW\
            --lr 5e-5\
            --weight-decay 1e-8\
            --scheduler CosineAnnealingLR\
            --scheduler_kwargs T_max=$epochs\
            --dataset iwildcam\
            --root_dir data/iwildcam\
            --train_split train\
            --eval_split id_val val all_val id_test test\
            --model_selection_split all_val\
            --model_selection_metric macro_f1\
            --metric acc loss macro_f1\
            --distributed
"""

# count = 2
# while count > 1:
# ret = subprocess.getoutput('ps -p 1930148 | wc')
# count = int(re.split(r'\s+', ret)[1])
# time.sleep(10)

for root, dirs, files in os.walk("weights/iwildcam/ablation"):
    for filename in files:
        if filename.endswith(".npy"):
            for seed in range(1, 2):
                # command = cmd.format(weights=os.path.join(root, filename), seed=seed)
                # os.system(command)
                # selection(path='results', max_iter=500, name_filter=f'results/report/lambda_ablation/{filename}',\
                # selectors=['id_val acc', 'val acc', 'all_val acc', 'test acc', 'id_val macro_f1', 'val macro_f1', 'all_val macro_f1', 'test macro_f1'])
                pass
