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

Download model weights:
- [CLaMP 2 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth)
- [M3 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth)

Put the models under the ```clamp2/```folder.
