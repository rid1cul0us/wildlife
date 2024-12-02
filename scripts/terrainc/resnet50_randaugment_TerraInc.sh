export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
GPU_COUNT=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=`seq -s, 0 $((GPU_COUNT-1))`
torchrun --nproc_per_node=$GPU_COUNT\
            main.py\
            --arch resnet50\
            --seed $1\
            --method ERM\
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
            --save_each_epoch \
            --distributed
