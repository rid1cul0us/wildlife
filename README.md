### Dataset
<ul>
  <li><a href="https://worksheets.codalab.org/rest/bundles/0x6313da2b204647e79a14b468131fcd64/contents/blob/">IWildCam</a></li>
  <li><a href="http://lila.science/datasets/caltech-camera-traps">TerraInc</a></li>
</ul>

---
### Requirements
torch>=2.0.1

CUDA 11.7

other versions may be ok.

---
### Train and Test
Refer to script files under ./scripts
And ./args.py describes all arguments.

For example, ERM with ResNet50 :
```bash
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
            --scale 448 448\
            --train_transform randaugment_transform\
            --augment_kwargs num_ops=3 magnitude=9\
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
```

---

### Other
- for UserWarning: Grad strides do not match bucket view strides.

change `forward` function of class `torch.nn.modules.container.Sequential`
at `/home/anaconda3/envs/myenv/lib/python3.xx/site-packages/torch/nn/modules/container.py`
from 
```python
def forward(self, input):
    for module in self:
        input = module(input)
    return input
```
to
```python
def forward(self, input):
    for module in self:
        input = module(input).contiguous()
    return input
```
to avoid the warning
