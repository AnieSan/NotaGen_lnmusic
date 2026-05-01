"""One-off helper: rewrite absolute paths in dataset jsonl to repo-relative paths."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]  # notagen 仓库根
NEW_PREFIX = "NotaGen-main/data/augmented"


def _legacy_prefixes() -> list[str]:
    """仅由仓库根路径拼接，不写死盘符。"""
    candidates = [
        ROOT / "AUGMENTED_FOLDER",
        ROOT / "NotaGen-main" / "data" / "augmented" / "AUGMENTED_FOLDER",
    ]
    out: list[str] = []
    for p in candidates:
        try:
            out.append(str(p.resolve()).replace("\\", "/"))
        except OSError:
            out.append(str(p).replace("\\", "/"))
    return list(dict.fromkeys(out))


OLD = _legacy_prefixes()


def fix_path(p: str) -> str:
    if not p:
        return p
    norm = p.replace("\\", "/")
    for o in OLD:
        on = o.replace("\\", "/")
        if norm.startswith(on):
            rest = norm[len(on) :].lstrip("/")
            return f"{NEW_PREFIX}/{rest}" if rest else NEW_PREFIX
    if "AUGMENTED_FOLDER" in norm:
        return norm.split("AUGMENTED_FOLDER", 1)[-1].lstrip("/\\")
    return p


OUT_NAMES = {
    "AUGMENTED_FOLDER_train.jsonl": "dataset_train.jsonl",
    "AUGMENTED_FOLDER_eval.jsonl": "dataset_eval.jsonl",
    "AUGMENTED_FOLDER.jsonl": "dataset_all.jsonl",
}


def process(name: str):
    src = ROOT / name
    if not src.exists():
        print("skip missing", src)
        return
    out = HERE / OUT_NAMES.get(name, "dataset_" + name)
    lines_out = []
    with open(src, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "path" in obj:
                obj["path"] = fix_path(obj["path"])
            lines_out.append(json.dumps(obj, ensure_ascii=False) + "\n")
    with open(out, "w", encoding="utf-8") as w:
        w.writelines(lines_out)
    print("wrote", out, "lines", len(lines_out))


if __name__ == "__main__":
    for fn in (
        "AUGMENTED_FOLDER_train.jsonl",
        "AUGMENTED_FOLDER_eval.jsonl",
        "AUGMENTED_FOLDER.jsonl",
    ):
        process(fn)
