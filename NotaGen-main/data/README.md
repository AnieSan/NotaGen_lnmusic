## Data Pre-processing

**本副本目录约定（自动生成路径）**：`2_data_preprocess.py` 中 `ORI_FOLDER` / `INTERLEAVED_FOLDER` / `AUGMENTED_FOLDER` 已默认指向本目录下的 `interim/abc_output`、`interleaved`、`augmented`；索引输出到 `indices/dataset_*.jsonl`。随机划分使用与 **`finetune/config.py` 中 `RANDOM_SEED`** 相同的种子；输入文件按名字排序后再划分，保证可复现。

**固定验证集**：将 `indices/eval_paths_fixed.example.txt` 复制为 **`indices/eval_paths_fixed.txt`**，每行写一个样本的 `path`（与 jsonl 中完全一致）。存在该文件时，所列样本只进入 **`dataset_eval.jsonl`**，其余进入 **`dataset_train.jsonl`**（不再按比例随机划）。

### Convert from MusicXML

- Navigate to ```NotaGen-main/data/```
- ```1_batch_xml2abc.py``` 默认路径（相对本目录）：输入 **`raw/xml/`**，输出 **`interim/abc_output/`**，日志 **`logs/xml2abc_error_log.txt`**。若需修改，编辑脚本顶部 ```Path``` 变量。
  ```
  python 1_batch_xml2abc.py
  ```
  This will conver the MusicXML files into standard ABC notation files.
- Modify the ```ORI_FOLDER```, ```INTERLEAVED_FOLDER```, ```AUGMENTED_FOLDER```, and ```EVAL_SPLIT``` in ```2_data_preprocess.py```:
  
  ```python
  ORI_FOLDER = ''  # Folder containing standard ABC notation files
  INTERLEAVED_FOLDER = ''   # Output interleaved ABC notation files that are compatible with CLaMP 2 to this folder
  AUGMENTED_FOLDER = ''   # On the basis of interleaved ABC, output key-augmented and rest-omitted files that are compatible with NotaGen to this folder
  EVAL_SPLIT = 0.1    # Evaluation data ratio
  ```
  then run this script:
  ```
  python 2_data_preprocess.py
  ```
  - The script will convert the standard ABC to interleaved ABC, which is compatible with CLaMP 2. The files will be under ```INTERLEAVED_FOLDER```.

  - This script will make 15 key signature folders under the ```AUGMENTED_FOLDER```, and output interleaved ABC notation files with rest bars omitted. This is the data representation that NotaGen adopts.
  
  - This script will also generate data index files for training NotaGen. It will randomly split train and eval sets according to the proportion ```EVAL_SPLIT``` defines. The index files will be named as ```{AUGMENTED_FOLDER}_train.jsonl``` and ```{AUGMENTED_FOLDER}_eval.jsonl```.

## Data Post-processing

### Preview Sheets in ABC Notation

We recommend [EasyABC](https://sourceforge.net/projects/easyabc/), a nice software for ABC Notation previewing, composing and editing.

It's needed to add a line "X:1" before each piece to present the score image in EasyABC :D

### Convert to MusicXML

- Go to ```NotaGen-main/data/```
- ```3_batch_abc2xml.py``` 默认路径：输入 **`../runs/exports/abc_generated/`**，输出 **`../runs/exports/xml_generated/`**，日志 **`logs/abc2xml_error_log.txt`**。
  ```
  python 3_batch_abc2xml.py
  ```
  This will conver the standard/interleaved ABC notation files into MusicXML files.
