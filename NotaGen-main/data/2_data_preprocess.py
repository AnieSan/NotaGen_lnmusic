from __future__ import annotations

from pathlib import Path
import sys
from typing import Optional, Set

# 所有路径相对本仓库：以 NotaGen-main 为工程根，数据在 NotaGen-main/data 下
_DATA_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _DATA_DIR.parents[1]  # notagen（仓库根）

_FINETUNE_DIR = _DATA_DIR.parent / "finetune"
if str(_FINETUNE_DIR.resolve()) not in sys.path:
    sys.path.insert(0, str(_FINETUNE_DIR.resolve()))
from config import RANDOM_SEED

ORI_FOLDER = str(_DATA_DIR / "interim" / "abc_output")
INTERLEAVED_FOLDER = str(_DATA_DIR / "interleaved")
AUGMENTED_FOLDER = str(_DATA_DIR / "augmented")
EVAL_SPLIT = 0.1  # 验证集比例（当未使用固定验证集清单时）
# 固定验证集：若存在下列文件，则其中列出的 path（与 jsonl 中完全一致）仅进入 eval，其余进 train
_EVAL_FIXED_FILE = _DATA_DIR / "indices" / "eval_paths_fixed.txt"

import os
import re
import json
import shutil
import random
from tqdm import tqdm
from abctoolkit.utils import (
    remove_information_field, 
    remove_bar_no_annotations, 
    Quote_re, 
    Barlines,
    extract_metadata_and_parts, 
    extract_global_and_local_metadata,
    extract_barline_and_bartext_dict)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated
from abctoolkit.transpose import Key2index, transpose_an_abc_text

os.makedirs(INTERLEAVED_FOLDER, exist_ok=True)
os.makedirs(AUGMENTED_FOLDER, exist_ok=True)
for key in Key2index.keys():
    key_folder = os.path.join(AUGMENTED_FOLDER, key)
    os.makedirs(key_folder, exist_ok=True)


def abc_preprocess_pipeline(abc_path):

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    # delete blank lines
    abc_lines = [line for line in abc_lines if line.strip() != '']

    # unidecode
    abc_lines = unidecode_abc_lines(abc_lines)

    # clean information field
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

    # delete bar number annotations
    abc_lines = remove_bar_no_annotations(abc_lines)

    # delete \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # delete text annotations with quotes
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # check bar alignment
    try:
        _, bar_no_equal_flag, _ = check_alignment_unrotated(abc_lines)
        if not bar_no_equal_flag:
            print(abc_path, 'Unequal bar number')
            raise Exception
    except:
        raise Exception

    # deal with text annotations: remove too long text annotations; remove consecutive non-alphabet/number characters
    for i, line in enumerate(abc_lines):
        quote_matches = re.findall(r'"[^"]*"', line)
        for match in quote_matches:
            if match == '""':
                line = line.replace(match, '')
            if match[1] in ['^', '_']:
                sub_string = match
                pattern = r'([^a-zA-Z0-9])\1+'
                sub_string = re.sub(pattern, r'\1', sub_string)
                if len(sub_string) <= 40:
                    line = line.replace(match, sub_string)
                else:
                    line = line.replace(match, '')
        abc_lines[i] = line

    abc_name = os.path.splitext(os.path.split(abc_path)[-1])[0]

    # transpose
    metadata_lines, part_text_dict = extract_metadata_and_parts(abc_lines)
    global_metadata_dict, local_metadata_dict = extract_global_and_local_metadata(metadata_lines)
    if global_metadata_dict['K'][0] == 'none':
        global_metadata_dict['K'][0] = 'C'
    ori_key = global_metadata_dict['K'][0]

    interleaved_abc = rotate_abc(abc_lines)
    interleaved_path = os.path.join(INTERLEAVED_FOLDER, abc_name + '.abc')
    with open(interleaved_path, 'w') as w:
        w.writelines(interleaved_abc)

    for key in Key2index.keys():
        transposed_abc_text = transpose_an_abc_text(abc_lines, key)
        transposed_abc_lines = transposed_abc_text.split('\n')
        transposed_abc_lines = list(filter(None, transposed_abc_lines))
        transposed_abc_lines = [line + '\n' for line in transposed_abc_lines]

        # rest reduction
        metadata_lines, prefix_dict, left_barline_dict, bar_text_dict, right_barline_dict = \
            extract_barline_and_bartext_dict(transposed_abc_lines)
        reduced_abc_lines = metadata_lines
        for i in range(len(bar_text_dict['V:1'])):
            line = ''
            for symbol in prefix_dict.keys():
                valid_flag = False
                for char in bar_text_dict[symbol][i]:
                    if char.isalpha() and not char in ['Z', 'z', 'X', 'x']:
                        valid_flag = True
                        break
                if valid_flag:
                    if i == 0:
                        part_patch = '[' + symbol + ']' + prefix_dict[symbol] + left_barline_dict[symbol][0] + bar_text_dict[symbol][0] + right_barline_dict[symbol][0]
                    else:
                        part_patch = '[' + symbol + ']' + bar_text_dict[symbol][i] + right_barline_dict[symbol][i]
                    line += part_patch
            line += '\n'
            reduced_abc_lines.append(line)
            
            reduced_abc_name = abc_name + '_' + key
            reduced_abc_path = os.path.join(AUGMENTED_FOLDER, key, reduced_abc_name + '.abc')
        
            with open(reduced_abc_path, 'w', encoding='utf-8') as w:
                w.writelines(reduced_abc_lines)

    return abc_name, ori_key





def _load_fixed_eval_paths() -> Optional[Set[str]]:
    if not _EVAL_FIXED_FILE.is_file():
        return None
    out = set()
    for line in _EVAL_FIXED_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        out.add(line.replace("\\", "/"))
    return out if out else None


if __name__ == '__main__':
    data = []
    file_list = sorted(
        f for f in os.listdir(ORI_FOLDER) if f.lower().endswith(".abc")
    )
    for file in tqdm(file_list):
        ori_abc_path = os.path.join(ORI_FOLDER, file)
        try:
            abc_name, ori_key = abc_preprocess_pipeline(ori_abc_path)
        except:
            print(ori_abc_path, 'failed to pre-process.')
            continue

        aug_base = os.path.join(AUGMENTED_FOLDER, abc_name)
        try:
            path_for_index = str(Path(aug_base).resolve().relative_to(_REPO_ROOT)).replace("\\", "/")
        except ValueError:
            path_for_index = aug_base.replace("\\", "/")
        data.append({"path": path_for_index, "key": ori_key})

    data.sort(key=lambda x: x["path"])
    fixed_eval = _load_fixed_eval_paths()
    if fixed_eval is not None:
        all_paths = {d["path"] for d in data}
        missing = fixed_eval - all_paths
        if missing:
            print(
                "警告: eval_paths_fixed.txt 中有下列 path 未在本次预处理结果中找到（请核对路径是否完全一致）:",
                *sorted(missing)[:20],
                sep="\n  ",
            )
        train_data = [d for d in data if d["path"] not in fixed_eval]
        eval_data = [d for d in data if d["path"] in fixed_eval]
        if not eval_data:
            raise SystemExit("eval_paths_fixed.txt 已指定但与数据无交集，请检查 path 是否与 jsonl 字段一致。")
        if not train_data:
            raise SystemExit("固定验证集占满全部样本，train 为空。")
    else:
        rng = random.Random(RANDOM_SEED)
        shuffled = list(data)
        rng.shuffle(shuffled)
        n_eval = max(1, int(EVAL_SPLIT * len(shuffled))) if len(shuffled) > 1 else 0
        eval_data = shuffled[:n_eval]
        train_data = shuffled[n_eval:]

    _indices = _DATA_DIR / "indices"
    _indices.mkdir(parents=True, exist_ok=True)
    data_index_path = str(_indices / "dataset_all.jsonl")
    eval_index_path = str(_indices / "dataset_eval.jsonl")
    train_index_path = str(_indices / "dataset_train.jsonl")


    with open(data_index_path, 'w', encoding='utf-8') as w:
        for d in data:
            w.write(json.dumps(d) + '\n')
    with open(eval_index_path, 'w', encoding='utf-8') as w:
        for d in eval_data:
            w.write(json.dumps(d) + '\n')
    with open(train_index_path, 'w', encoding='utf-8') as w:
        for d in train_data:
            w.write(json.dumps(d) + '\n')

    

