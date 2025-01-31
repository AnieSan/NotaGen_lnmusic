## Links
- [CLaMP 2 Paper](https://arxiv.org/pdf/2410.13267)
- [CLaMP 2 Code](https://github.com/sanderwood/clamp2)

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

### Convert from MusicXML

- Go to the data folder ```cd data/```
- Change the ```ORI_FOLDER``` and ```DES_FOLDER``` in ```1_batch_xml2abc.py```, then run this script:
  ```
  python 1_batch_xml2abc.py
  ```
  This will conver the MusicXML files into standard ABC notation files, like this:
  ```plain
  %%score { 1 | 2 }
  L:1/8
  Q:1/4=120
  M:3/4
  K:G
  V:1 treble nm="Piano" snm="Pno."
  V:2 bass
  V:1
  !mf!"^Allegro" d2 (GA Bc | d2) .G2 .G2 |]
  V:2
  [G,B,D]4 A,2 | B,6 |]
  ```
- Change the ```ORI_FOLDER```, ```DES_FOLDER```, and ```EVAL_SPLIT``` in ```2_data_preprocess.py```, then run this script:
  ```
  python 2_data_preprocess.py
  ```
  This script will make 15 key signature folders under the ```DES_FOLDER```, and output interleaved ABC notation files with rest bars omitted.
  The interleaved ABC notation is like this:
  ```plain
  %%score { 1 | 2 }
  L:1/8
  Q:1/4=120
  M:3/4
  K:G
  V:1 treble nm="Piano" snm="Pno."
  V:2 bass
  [V:1]!mf!"^Allegro" d2 (GA Bc|[V:2][G,B,D]4 A,2|
  [V:1]d2) .G2 .G2|][V:2]B,6|]
  ```

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
