# 仓库目录说明（以 NotaGen-main 为主）

## 工程入口

所有训练、数据预处理、推理请以 **`NotaGen-main/`** 为准：

- 微调 / LoRA：`NotaGen-main/finetune/`
- 数据脚本与数据目录：`NotaGen-main/data/`
- 预训练 / 官方其它模块：`NotaGen-main/pretrain/` 等

在 **`NotaGen-main/finetune`** 下运行 Python，以便 `config.py` 中的相对路径正确。

## 数据目录（已迁入 NotaGen-main/data）

**Git 约定**：仓库内只保留上述子目录结构与预处理脚本；**具体 MusicXML / ABC / `indices/*.jsonl` / 转换日志等均在 `.gitignore` 中排除**，需在本地放置语料并运行 `1_batch_xml2abc.py`、`2_data_preprocess.py` 生成。

| 路径 | 含义 |
|------|------|
| `data/raw/xml/` | 原始 MusicXML |
| `data/interim/abc_output/` | 标准 ABC（xml2abc 等中间产物） |
| `data/interleaved/` | 交错 ABC |
| `data/augmented/` | 增强数据（按调号子目录） |
| `data/indices/dataset_*.jsonl` | 训练/验证/全量索引 |
| `data/indices/eval_paths_fixed.txt` | （可选）固定验证集：每行一条与 jsonl 一致的 `path` |
| `data/logs/` | 转换错误日志（若有） |

## 实验产出

| 路径 | 含义 |
|------|------|
| `NotaGen-main/weights/` | 预训练或合并后的 `.pth`（请自行放入权重文件） |
| `NotaGen-main/runs/default/` | 全量微调日志与 `config` 中 `WEIGHTS_PATH` 保存位置 |
| `NotaGen-main/runs/exports/` | 生成样例（原 `abc_generated` / `xml_generated`） |
| `NotaGen-main/runs/inference/` | 批量推理输出（`inference/config.py` 中 `ORIGINAL_OUTPUT_FOLDER` 等） |

## 根目录遗留

- **`_archive_pre_layout_20260412/`**：整理前的旧版 `AUGMENTED_FOLDER_*.jsonl` 备份。
- **`_duplicate_root_modules_*.zip`**：与 `NotaGen-main` 重复的根目录模块（`clamp2`、`finetune`、`gradio`、`inference`、`notebook`、`pretrain`、`RL`、`data`）已打包备份后**删除**；恢复时可解压到仓库根目录（不推荐，请以 `NotaGen-main` 为准）。

## 重新生成索引

在仓库根目录执行（或于 `NotaGen-main/data` 下）：

```bash
cd NotaGen-main/data
python 2_data_preprocess.py
```

索引将写入 `data/indices/dataset_train.jsonl` 等；`path` 字段为相对仓库根的路径（`NotaGen-main/data/augmented/...`）。

## Windows：Cursor 里使用 conda

已修正 **`Documents\WindowsPowerShell\profile.ps1`**：原先 `conda init` 指向已失效的临时目录 `...\Temp\_MEI*\conda.exe`，导致每次新开终端都无法加载 `conda`。现已改为固定使用 **`D:\miniconda\Scripts\conda.exe`** 的 hook。

若执行策略仍拦截脚本，可在 PowerShell 执行：`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force`，然后 **关掉并重开 Cursor 终端**，再运行 `conda activate notagen`。
