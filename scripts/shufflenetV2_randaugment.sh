export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
GPU_COUNT=5
export CUDA_VISIBLE_DEVICES=5,6,7,8,9
torchrun --nproc_per_node=$GPU_COUNT\
            --master-port 29501\
            main.py\
            --result-dir results/\
            --arch shufflenet\
            --seed $1\
            --method ERM\
            --epochs $epochs\
            --batch-size 128\
            --scale 224 336\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss CrossEntropyLoss\
            --optimizer AdamW\
            --lr 5e-5\
            --weight-decay 1e-8\
            --scheduler CosineAnnealingLR\
            --scheduler_kwargs T_max=$epochs\
            --dataset iwildcam-folder\
            --train_split train\
            --eval_split id_val val all_val test\
            --model_selection_split all_val\
            --model_selection_metric macro_f1\
            --metric acc loss macro_f1\
            --save_each_epoch \
            --distributed
