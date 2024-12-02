import os


if __name__ == "__main__":
    lr2eps = {
        "5e-4": "20",
        "1e-4": "20",
        "9e-5": "20",
        "8e-5": "20",
        "7e-5": "20",
        "6e-5": "20",
        "5e-5": "20",
        "4e-5": "25",
        "3e-5": "30",
        "2e-5": "35",
        "1e-5": "40",
        "5e-6": "45",
    }

    for lr, epochs in lr2eps.items():
        os.system(
            f"""
    export PYTHONPATH=`pwd`
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4

    echo "lr {lr} epoch {epochs}"

    torchrun --nproc_per_node=5\
                main.py\
                --method ERM\
                --arch convnext_tiny\
                --seed 1\
                --epochs {epochs}\
                --batch-size 48\
                --eval_batch_size 96\
                --scale 448 448\
                --train_transform randaugment_transform\
                --augment_kwargs num_ops=3 magnitude=9\
                --loss CrossEntropyLoss\
                --optimizer AdamW\
                --lr {lr}\
                --weight-decay 1e-8\
                --scheduler CosineAnnealingLR\
                --scheduler_kwargs T_max={epochs}\
                --dataset iwildcam\
                --root_dir data/iwildcam\
                --train_split train\
                --eval_split all_val id_test test\
                --model_selection_split all_val\
                --model_selection_metric macro_f1\
                --metric acc loss macro_f1\
                --distributed
    """
        )
