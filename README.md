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