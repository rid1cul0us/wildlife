export PYTHONPATH=`pwd`
epochs=20
GPU_COUNT=5
# export CUDA_VISIBLE_DEVICES=`seq -s, 0 $((GPU_COUNT-1)) | paste -sd ''`
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=0,1,2,5,6
torchrun --nproc_per_node=$GPU_COUNT\
            main.py\
            --method PU\
            --result-dir results/report/bw/PU/vit\
            --arch swinv2_vit_tiny\
            --weights weights/iwildcam/global/shifted_softmax_loss_weight_0.4.npy\
            --seed $1\
            --epochs $epochs\
            --batch-size 36\
            --scale 448 448\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss CrossEntropyLoss\
            --optimizer AdamW\
            --lr 3e-5\
            --weight-decay 0.05\
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
