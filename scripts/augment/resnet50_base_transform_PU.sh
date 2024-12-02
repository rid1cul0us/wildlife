export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
GPU_COUNT=5 #`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=5,6,7,8,9
torchrun --nproc_per_node=$GPU_COUNT\
            --master_port=25689\
            main.py\
            --arch resnet50\
            --result-dir results/report/bw/PU/base_augment\
            --weights weights/iwildcam/global/original_loss_weight.npy\
            --seed $1\
            --method PU\
            --epochs $epochs\
            --batch-size 48\
            --scale 448 448\
            --train_transform base_transform\
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
            --save_each_epoch \
            --distributed