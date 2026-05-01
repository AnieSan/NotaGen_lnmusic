from pathlib import Path

# 相对本文件所在目录 NotaGen-main/data/
_DATA_DIR = Path(__file__).resolve().parent
ORI_FOLDER = str(_DATA_DIR / "raw" / "xml")
DES_FOLDER = str(_DATA_DIR / "interim" / "abc_output")
_LOG_DIR = _DATA_DIR / "logs"
_XML2ABC = _DATA_DIR / "xml2abc.py"

import os
import sys
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def _decode_subprocess_stdout(raw: bytes) -> str:
    """Windows 上 xml2abc 往往按系统代码页（如 gbk）写 stdout，不能假定 utf-8。"""
    if not raw:
        return ""
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936", "mbcs"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def convert_xml2abc(file_list):
    py = sys.executable
    cmd_base = f'"{py}" "{_XML2ABC}" -d 8 -c 6 -x '
    err_log = _LOG_DIR / "xml2abc_error_log.txt"
    for file in tqdm(file_list):
        filename = os.path.basename(file)
        os.makedirs(DES_FOLDER, exist_ok=True)

        try:
            p = subprocess.Popen(
                cmd_base + '"' + file + '"',
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
            )
            out_b, err_b = p.communicate()
            output = _decode_subprocess_stdout(out_b)

            if not output.strip():
                err_txt = _decode_subprocess_stdout(err_b).strip() if err_b else ""
                with open(err_log, "a", encoding="utf-8") as f:
                    f.write(file + (f" | stderr: {err_txt[:500]}" if err_txt else " (empty stdout)") + "\n")
                continue

            out_path = os.path.join(DES_FOLDER, filename.rsplit(".", 1)[0] + ".abc")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(output)
        except Exception as e:
            with open(err_log, "a", encoding="utf-8") as f:
                f.write(file + " " + str(e) + "\n")


if __name__ == "__main__":
    import sys

    _fin = Path(__file__).resolve().parent.parent / "finetune"
    if str(_fin) not in sys.path:
        sys.path.insert(0, str(_fin))
    from config import RANDOM_SEED

    file_list = []
    _LOG_DIR.mkdir(parents=True, exist_ok=True)

    for root, dirs, files in os.walk(os.path.abspath(ORI_FOLDER)):
        for file in sorted(files):
            if file.endswith((".mxl", ".xml", ".musicxml")):
                filename = os.path.join(root, file).replace("\\", "/")
                file_list.append(filename)

    random.seed(RANDOM_SEED)
    random.shuffle(file_list)
    num_files = len(file_list)
    num_processes = os.cpu_count()
    file_lists = [file_list[i::num_processes] for i in range(num_processes)]

    with Pool(processes=num_processes) as pool:
        pool.map(convert_xml2abc, file_lists)
