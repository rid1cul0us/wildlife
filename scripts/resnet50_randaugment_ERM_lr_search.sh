export PYTHONPATH=`pwd`
epochs=20
GPU_COUNT=`nvidia-smi -L | wc -l`
# GPU_COUNT=3
# export CUDA_VISIBLE_DEVICES=0,1,2
export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# for lr in "5e-3" "1e-3" "5e-4" "1e-4" "9e-5" "8e-5" "7e-5" "6e-5" "5e-5" "4e-5" "3e-5" "2e-5" "1e-5" "5e-6"
for lr in "5e-4" "1e-4" "9e-5" "8e-5" "7e-5" "6e-5" "5e-5" "4e-5" "3e-5" "2e-5" "1e-5" "5e-6"
do

echo "lr $lr"

torchrun --nproc_per_node=5\
            main.py\
            --method ERM\
            --arch resnet50\
            --seed 1\
            --epochs $epochs\
            --batch-size 48\
            --eval_batch_size 96\
            --scale 448 448\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss CrossEntropyLoss\
            --optimizer AdamW\
            --lr $lr\
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

done