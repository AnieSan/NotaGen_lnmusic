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
  This will conver the MusicXML files into standard ABC notation files.
- Change the ```ORI_FOLDER```, ```INTERLEAVED_FOLDER```, ```AUGMENTED_FOLDER```, and ```EVAL_SPLIT``` in ```2_data_preprocess.py```:
  
  ```python
  ORI_FOLDER = ''  # Folder containing standard ABC notation files
  INTERLEAVED_FOLDER = ''   # Output interleaved ABC notation files that are compatible with CLaMP 2 to this folder
  AUGMENTED_FOLDER = ''   # On the basis of interleaved ABC, output key-augmented and rest-omitted files that are compatible with NotaGen to this folder
  EVAL_SPLIT = 0.1    # The ratio of eval data 
  ```
  then run this script:
  ```
  python 2_data_preprocess.py
  ```
  - The script will convert the standard ABC to interleaved ABC, which is compatible with CLaMP 2. The files will be under ```INTERLEAVED_FOLDER```.

  - This script will make 15 key signature folders under the ```AUGMENTED_FOLDER```, and output interleaved ABC notation files with rest bars omitted. This is the data representation that NotaGen adopts.
  
  - This script will also generate data index files for training NotaGen. It will randomly split train and eval sets according to the proportion ```EVAL_SPLIT``` defines. The index files will be named as ```{AUGMENTED_FOLDER}_train.jsonl``` and ```{AUGMENTED_FOLDER}_eval.jsonl```.

### Data Examples

To illustrate the specific data format, we provide a subset of the [OpenScore Lieder](https://github.com/OpenScore/Lieder) data, which includes interleaved ABC folders, augmented ABC folders, as well as data index files for training and evaluation. You can download it here and unzip it under ```data/```.

In the instructions of Fine-tuning and Reinforcement Learning below, we will use this dataset as an example of our implementation. It won't include the "period-composer-instrumentation" conditioning, just for showing how to adapt the pretrained NotaGen to a specific music style.


## Pre-train
If you want to use your data to pre-train a blank NotaGen model, please preprocess the data and generate the data index files following the instructions above. Then modify the parameters in ```pretrain/config.py```.

Use this command for pre-training:
```
cd pretrain/
accelerate launch --multi_gpu --mixed_precision fp16 train-gen.py
```
We also provide pre-trained ckpts of different scales:
|  Models         |  Parameters  |
|  ----           |  ----        |
|  NotaGen-Small  | 110M         |
|  NotaGen-Medium | 244M         |
|  NotaGen-Large  | 516M         |


## Fine-tune
- In ```finetune/config.py```:
  - Change the ```DATA_TRAIN_INDEX_PATH``` and ```DATA_EVAL_INDEX_PATH```:
  ```python
  # Configuration for the data
  DATA_TRAIN_INDEX_PATH = "../data/openscorelieder_augmented_train.jsonl" 
  DATA_EVAL_INDEX_PATH  = "../data/openscorelieder_augmented_train.jsonl"
  ```
  - Change the ```PRETRAINED_PATH``` to the pre-trained NotaGen weights:
  ```python
  PRETRAINED_PATH = "../pretrain/weights_notagen_pretrain_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_0.0001_batch_4.pth"
  ```
  - ```EXP_TAG``` is for differentiating the models. It will be integrated into the ckpt's name. We can set it to ```openscorelieder```.
  - You can also modify other parameters like the learning rate.
- Use this command for fine-tuning
  ```
  cd finetune/
  python train-gen.py
  ```



## Reinforcement Learning (CLaMP-DPO)
### CLaMP 2 Setup

Download model weights and put them under the ```clamp2/```folder:
- [CLaMP 2 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_clamp2_h_size_768_lr_5e-05_batch_128_scale_1_t_length_128_t_model_FacebookAI_xlm-roberta-base_t_dropout_True_m3_True.pth)
- [M3 Model Weights](https://huggingface.co/sander-wood/clamp2/blob/main/weights_m3_p_size_64_p_length_512_t_layers_3_p_layers_12_h_size_768_lr_0.0001_batch_16_mask_0.45.pth)

### Extract Ground Truth Features
Change ```input_dir``` and ```output_dir``` in ```clamp2/extract_clamp2.py```:
```python
input_dir = '../data/openscorelieder_interleaved'  # interleaved abc folder
output_dir = 'feature/openscorelieder_interleaved'  # feature folder
```
Extract the features:
```
cd clamp2/
python extract_clamp2.py
```

### CLaMP-DPO

Here we give an example of an iteration of CLaMP-DPO from the initial fine-tuned model.

#### Inference
- Change the ```INFERENCE_WEIGHTS_PATH``` to path of the fine-tuned weights and ```NUM_SAMPLES``` to generate in ```inference/config.py```:
  ```python
    INFERENCE_WEIGHTS_PATH = '../finetune/weights_notagen_openscorelider_p_size_16_p_length_1024_p_layers_20_c_layers_6_h_size_1280_lr_1e-05_batch_1.pth'              
    NUM_SAMPLES = 1000                                               
  ```
- Inference:
  ```
  cd inference/
  python inference.py
  ```
  This will generate an ```output/```folder with two subfolders: ```original``` and ```interleaved```. The ```original/``` subdirectory stores the raw inference outputs from the model, while the ```interleaved/``` subdirectory contains data post-processed with rest measure completion, compatible with CLaMP 2. Each of these subdirectories will contain a model-specific folder, named as a combination of the model's name and its sampling parameters.
  
   
