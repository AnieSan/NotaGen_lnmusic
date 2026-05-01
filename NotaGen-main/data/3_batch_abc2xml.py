from pathlib import Path

# 相对 NotaGen-main：ABC 输入与 XML 输出放在 runs/exports/（与 PROJECT_LAYOUT 一致）
_DATA_DIR = Path(__file__).resolve().parent
_NG_ROOT = _DATA_DIR.parent
ORI_FOLDER = str(_NG_ROOT / "runs" / "exports" / "abc_generated")
DES_FOLDER = str(_NG_ROOT / "runs" / "exports" / "xml_generated")
_LOG_DIR = _DATA_DIR / "logs"
_ABC2XML = _DATA_DIR / "abc2xml.py"

import os
import sys
import math
import random
import subprocess
from tqdm import tqdm
from multiprocessing import Pool


def _decode_subprocess_stdout(raw: bytes) -> str:
    if not raw:
        return ""
    for enc in ("utf-8", "utf-8-sig", "gbk", "cp936", "mbcs"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def convert_abc2xml(file_list):
    py = sys.executable
    cmd_base = f'"{py}" "{_ABC2XML}" '
    err_log = _LOG_DIR / "abc2xml_error_log.txt"
    for file in tqdm(file_list):
        filename = file.split("/")[-1]
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
            output_path = f"{DES_FOLDER}/" + ".".join(filename.split(".")[:-1]) + ".xml"
            with open(output_path, "w", encoding="utf-8") as f:
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

    for root, dirs, files in os.walk(ORI_FOLDER):
        for file in sorted(files):
            if not file.endswith(".abc"):
                continue
            filename = os.path.join(root, file).replace("\\", "/")
            file_list.append(filename)

    file_lists = []
    random.seed(RANDOM_SEED)
    random.shuffle(file_list)
    for i in range(os.cpu_count()):
        start_idx = int(math.floor(i * len(file_list) / os.cpu_count()))
        end_idx = int(math.floor((i + 1) * len(file_list) / os.cpu_count()))
        file_lists.append(file_list[start_idx:end_idx])

    pool = Pool(processes=os.cpu_count())
    pool.map(convert_abc2xml, file_lists)
