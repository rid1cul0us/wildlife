# reference https://github.com/qiqi-helloworld/ABSGD/tree/main/wilds-competition

export PYTHONPATH=$PYTHONPATH:`pwd`
epochs=20
sc_gamma=0.5
# GPU_COUNT=`nvidia-smi -L | wc -l`
export CUDA_VISIBLE_DEVICES=0,1,2,3,4
torchrun --nproc_per_node=$GPU_COUNT\
            main_ABSGD.py\
            --arch resnet50\
            --seed $1\
            --method ERM\
            --epochs $epochs\
            --batch-size 48\
            --scale 448 448\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
            --loss ABSGD\
            --loss_kwargs loss_type='CE' abAlpha=1\
            --optimizer AdamW\
            --lr 5e-5\
            --weight-decay 1e-8\
            --scheduler StepLR\
            --scheduler_kwargs step_size=$(($epochs/2)) gamma=$sc_gamma\
            --dataset iwildcam\
            --root_dir data/iwildcam\
            --train_split train\
            --eval_split id_val val all_val id_test test\
            --model_selection_split all_val\
            --model_selection_metric macro_f1\
            --metric acc loss macro_f1\
            --save_each_epoch \
            --distributed
