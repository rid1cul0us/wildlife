export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
GPU_COUNT=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=`seq -s, 0 $((GPU_COUNT-1))`
p=0.8
q=2.0
eps=1e-2
torchrun --nproc_per_node=$GPU_COUNT\
            main.py\
            --arch resnet50\
            --seed $1\
            --method ERM\
            --result-dir results/report/Seesaw/p_${p}_q_${q}_eps_${eps} \
            --epochs $epochs\
            --batch-size 48\
            --scale 448 448\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss SeesawLoss\
            --loss_kwargs p=${p} q=${q} eps=${eps}\
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
