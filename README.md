## Environment Setup

```python
conda create --name notagen python=3.10
conda activate notagen
conda install pytorch==2.3.0 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install accelerate
pip install optimum
pip install -r requirements.txt
```

## Data Preprocessing

## Pretrain

```
accelerate launch --multi_gpu --mixed_precision fp16 train-gen.py
```

## Finetune

## Reinforcement Learning (CLaMP-DPO)
### CLaMP 2 Setup
```
git clone https://github.com/sanderwood/clamp2.git
```
