CUDA_VISIBLE_DEVICES=0,1 \
torchrun --nproc_per_node=2\
            --master_port=29501\
            /home/kafm/program/dl/demo/cct20/baseline.py\
            --arch=resnet18\
            --optim=Adam\
            --lr=0.00003\
            --momentum=0.5\
            --epochs=200\
            --batch-size=96\
            --scale=299\
            --fine-tuning=clf\
            # --model-path=/home/kafm/program/dl/demo/cct20/models/model_0.467.pth
            # --save_each_epoch