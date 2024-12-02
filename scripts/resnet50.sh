export PYTHONPATH=`pwd`
epochs=15
GPU_COUNT=`nvidia-smi -L | wc -l`
CUDA_VISIBLE_DEVICES=`seq -s, 0 $((GPU_COUNT-1)) | paste -sd ''`
torchrun --nproc_per_node=$GPU_COUNT\
            main.py\
            --arch resnet50\
            --seed $1\
            --epochs $epochs\
            --batch-size 64\
            --scale 448 448\
            --train_transform base_augment_transform\
            --loss CrossEntropyLoss\
            --optimizer AdamW\
            --lr 5e-5\
            --weight-decay 1e-8\
            --scheduler CosineAnnealingLR\
            --scheduler_kwargs T_max=$epochs\
            --dataset iwildcam\
            --root_dir data/iwildcam\
            --train_split train\
            --eval_split all_val id_test test\
            --model_selection_split all_val\
            --model_selection_metric macro_f1\
            --metric acc loss macro_f1\
            --distributed