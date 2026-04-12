import os
import gc
import time
import json
import math
import random
import argparse
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import GPT2Config, get_constant_schedule_with_warmup

from utils import Patchilizer, NotaGenLMHeadModel
from config import (
    PATCH_SIZE,
    PATCH_LENGTH,
    PATCH_NUM_LAYERS,
    CHAR_NUM_LAYERS,
    HIDDEN_SIZE,
    DATA_TRAIN_INDEX_PATH,
    DATA_EVAL_INDEX_PATH,
    PRETRAINED_PATH,
    BATCH_SIZE,
    LEARNING_RATE,
    NUM_EPOCHS,
    ACCUMULATION_STEPS,
    WANDB_LOGGING,
    WANDB_KEY,
    WANDB_NAME,
    WEIGHTS_PATH,
    LOGS_PATH,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model
except Exception as e:
    raise RuntimeError(
        "未检测到 peft。请先安装：pip install peft"
    ) from e


def ensure_forward_compat(model):
    """
    peft/transformers 会用 input_ids / attention_mask 关键字调用 forward。
    NotaGenLMHeadModel 在原仓库里 forward(patches, masks) 不兼容；这里在训练脚本里兜底适配，
    避免服务器端 utils.py 未同步导致的 keyword 参数报错。
    """
    import inspect
    import types

    sig = None
    try:
        sig = inspect.signature(model.forward)
    except Exception:
        sig = None

    if sig is not None and "input_ids" in sig.parameters:
        return model

    old_forward = model.forward

    def forward_compat(self, *args, **kwargs):
        # 1) 兼容 HF/peft 常见别名
        if "input_ids" in kwargs and "patches" not in kwargs:
            kwargs["patches"] = kwargs.pop("input_ids")
        if "attention_mask" in kwargs and "masks" not in kwargs:
            kwargs["masks"] = kwargs.pop("attention_mask")

        # 2) peft/transformers 可能还会传 inputs_embeds/labels/use_cache 等，
        # NotaGen 的 forward 不支持，直接忽略，确保只把 patches/masks 传下去。
        patches = kwargs.get("patches", None)
        masks = kwargs.get("masks", None)

        # 兼容位置参数调用：forward(patches, masks)
        if patches is None and len(args) >= 1:
            patches = args[0]
        if masks is None and len(args) >= 2:
            masks = args[1]

        return old_forward(patches, masks)

    model.forward = types.MethodType(forward_compat, model)
    return model


def split_data(data, eval_ratio: float = 0.01, seed: int = 0):
    if not data:
        return [], []
    rng = random.Random(seed)
    data = list(data)
    rng.shuffle(data)
    n_eval = max(1, int(len(data) * eval_ratio)) if len(data) > 1 else 0
    eval_data = data[:n_eval]
    train_data = data[n_eval:]
    return train_data, eval_data


def setup_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    global_rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="nccl")
    else:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    return world_size, global_rank, local_rank, device


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_unused_tensors(model, optimizer):
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class NotaGenDataset(Dataset):
    def __init__(
        self,
        filenames,
        patchilizer: Patchilizer,
        data_root: str = None,
        relative_base: str = "../../",
        strip_prefixes=None,
    ):
        self.filenames = filenames
        self.patchilizer = patchilizer
        self.data_root = data_root
        self.relative_base = relative_base
        self.strip_prefixes = strip_prefixes or []

    def _resolve_path(self, p: str) -> str:
        if p is None:
            return p
        p2_raw = str(p).strip()
        # 如果用户在 jsonl 里已经给了相对路径（./ 或 ../），则直接使用
        if p2_raw.startswith("./") or p2_raw.startswith("../"):
            return os.path.normpath(p2_raw)

        # jsonl 里可能是 Windows 路径（含盘符、反斜杠）；统一归一化
        p2 = p2_raw.replace("\\", "/")
        # 去掉类似 "E:" 这种盘符前缀，避免在 Linux 上被当成相对路径的一部分
        if len(p2) >= 2 and p2[1] == ":":
            p2 = p2[2:]
        p2 = p2.lstrip("/")

        # 去掉用户指定的路径前缀（用于把本地生成的 index 映射到服务器）
        for pref in self.strip_prefixes:
            if not pref:
                continue
            pref2 = str(pref).replace("\\", "/").lstrip("/")
            if p2.startswith(pref2):
                p2 = p2[len(pref2) :].lstrip("/")
                break

        # 统一映射为以 finetune/ 为运行目录的相对路径（例如 ../../）
        if self.relative_base:
            base = str(self.relative_base)
            if not (base.startswith("./") or base.startswith("../")):
                base = "../" + base if base.startswith("/") else "../../"
            p2 = os.path.join(base, p2)

        if self.data_root:
            return os.path.normpath(os.path.join(self.data_root, p2))
        return os.path.normpath(p2)

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filepath = self._resolve_path(self.filenames[idx]["path"])
        folder = os.path.dirname(filepath)
        name = os.path.split(filepath)[-1]

        # 原版 finetune/train-gen.py：按 key 子文件夹读取转调后的 abc
        # 这里保持一致：如果 index 里有 key 字段则按 key 读取，否则直接读 filepath。
        if "key" in self.filenames[idx]:
            from abctoolkit.transpose import Key2index, Key2Mode

            index2key = {index: key for key, index in Key2index.items() if index not in [1, 11]}
            mode2key = {mode: key for key, mode_list in Key2Mode.items() for mode in mode_list}

            ori_key = mode2key[self.filenames[idx]["key"]]
            ori_key_index = Key2index[ori_key]
            available_index = [(ori_key_index + offset) % 12 for offset in range(-3, 4)]
            index_prob = [1 / 16, 2 / 16, 3 / 16, 4 / 16, 3 / 16, 2 / 16, 1 / 16]
            index_prob_range = [0] + [sum(index_prob[0 : i + 1]) for i in range(len(index_prob))]
            random_number = random.random()
            des_key_index = available_index[3]
            for i in range(len(index_prob_range) - 1):
                if index_prob_range[i] <= random_number < index_prob_range[i + 1]:
                    des_key_index = available_index[i]

            if des_key_index == 1:
                des_key = "Db" if random.random() < 0.8 else "C#"
            elif des_key_index == 11:
                des_key = "B" if random.random() < 0.8 else "Cb"
            elif des_key_index == 6:
                des_key = "F#" if random.random() < 0.5 else "Gb"
            else:
                des_key = index2key[des_key_index]

            des_filepath = os.path.join(folder, des_key, name + "_" + des_key + ".abc")
            read_path = des_filepath if os.path.exists(des_filepath) else filepath
        else:
            read_path = filepath

        with open(read_path, "r", encoding="utf-8") as f:
            abc_text = f.read()

        file_bytes = self.patchilizer.encode_train(abc_text)
        file_masks = [1] * len(file_bytes)
        return torch.tensor(file_bytes, dtype=torch.long), torch.tensor(file_masks, dtype=torch.long)


def collate_batch(input_batches, device):
    input_patches, input_masks = zip(*input_batches)
    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)
    return input_patches.to(device), input_masks.to(device)


def split_into_minibatches(input_patches, input_masks, minibatch_size):
    minibatches = []
    for start_idx in range(0, len(input_patches), minibatch_size):
        end_idx = start_idx + minibatch_size
        minibatches.append((input_patches[start_idx:end_idx], input_masks[start_idx:end_idx]))
    return minibatches


def build_model(device):
    patch_config = GPT2Config(
        num_hidden_layers=PATCH_NUM_LAYERS,
        max_length=PATCH_LENGTH,
        max_position_embeddings=PATCH_LENGTH,
        n_embd=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=1,
    )
    char_config = GPT2Config(
        num_hidden_layers=CHAR_NUM_LAYERS,
        max_length=PATCH_SIZE + 1,
        max_position_embeddings=PATCH_SIZE + 1,
        hidden_size=HIDDEN_SIZE,
        num_attention_heads=HIDDEN_SIZE // 64,
        vocab_size=128,
    )
    model = NotaGenLMHeadModel(encoder_config=patch_config, decoder_config=char_config).to(device)
    return model


def load_pretrained_weights(model, ckpt_path: str, device):
    if not ckpt_path:
        return
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"预训练权重不存在：{ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint

    cpu_model = deepcopy(model).cpu()
    cpu_model.load_state_dict(state, strict=False)
    model.load_state_dict(cpu_model.state_dict(), strict=False)
    model.to(device)


def apply_lora(model, args):
    target_modules = [m.strip() for m in args.target_modules.split(",") if m.strip()]
    if not target_modules:
        raise ValueError("target_modules 不能为空，例如：c_attn,c_proj,c_fc")

    # NotaGenLMHeadModel 不是标准 HF CausalLM wrapper，这里用 FEATURE_EXTRACTION 更稳
    # LoRA 注入仍然会在子模块线性层上生效。
    lora_cfg = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        inference_mode=False,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


@torch.no_grad()
def evaluate(model, eval_loader, world_size, device, scaler_enabled: bool):
    model.eval()
    total_loss = 0.0
    it = 0
    for batch in tqdm(eval_loader, desc="eval", leave=False):
        input_patches, input_masks = batch
        with autocast(enabled=scaler_enabled):
            loss = model(input_ids=input_patches, attention_mask=input_masks).loss
        if world_size > 1:
            loss_t = loss.detach().float().unsqueeze(0)
            dist.reduce(loss_t, dst=0)
            loss_t = loss_t / world_size
            dist.broadcast(loss_t, src=0)
            loss = loss_t.squeeze(0)
        total_loss += float(loss.item())
        it += 1
    return total_loss / max(it, 1)


def main():
    parser = argparse.ArgumentParser(description="NotaGen LoRA 微调（基于 finetune/train-gen.py 改造）")
    parser.add_argument("--train_index", type=str, default=DATA_TRAIN_INDEX_PATH)
    parser.add_argument("--eval_index", type=str, default=DATA_EVAL_INDEX_PATH)
    parser.add_argument("--pretrained", type=str, default=PRETRAINED_PATH)
    parser.add_argument("--output_dir", type=str, default="lora_adapters")
    parser.add_argument(
        "--data_root",
        type=str,
        default=None,
        help="数据根目录：用于把 jsonl 中的 Windows/绝对路径映射到服务器路径（可选）",
    )
    parser.add_argument(
        "--relative_base",
        type=str,
        default="../../",
        help="把非相对路径统一映射为该相对前缀（默认 ../../，以 finetune/ 为运行目录）",
    )
    parser.add_argument(
        "--strip_prefix",
        type=str,
        default="aibianqu/notagen",
        help="去掉 jsonl 里 path 的前缀（逗号分隔多个）。默认 aibianqu/notagen",
    )

    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--accumulation_steps", type=int, default=ACCUMULATION_STEPS)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--target_modules", type=str, default="c_attn,c_proj,c_fc")
    parser.add_argument("--resume_adapter", type=str, default=None, help="从已保存的 LoRA adapter 目录继续训练（可选）")
    parser.add_argument("--fp16", action="store_true", help="启用 AMP（推荐在 CUDA 上）")
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default="checkpoints",
        help="完整 .pth 的保存目录（相对 finetune/）。训练结束会导出 merged_last.pth，并在最优时导出 merged_best.pth",
    )
    args = parser.parse_args()

    world_size, global_rank, local_rank, device = setup_distributed()
    set_seed(args.seed + global_rank)

    if global_rank == 0:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.ckpt_dir, exist_ok=True)

    patchilizer = Patchilizer()
    model = build_model(device)
    load_pretrained_weights(model, args.pretrained, device)
    model = ensure_forward_compat(model)

    model = apply_lora(model, args)

    if args.resume_adapter:
        # PeftModel 提供 from_pretrained，这里保持兼容：直接加载到当前 model
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.resume_adapter, is_trainable=True)

    # 只优化可训练参数（LoRA）
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)
    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=args.warmup_steps)

    scaler = GradScaler(enabled=args.fp16 and device.type == "cuda")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def unwrap(m):
        return m.module if hasattr(m, "module") else m

    # load data
    with open(args.train_index, "r", encoding="utf-8") as f:
        train_files = [json.loads(line) for line in f]
    with open(args.eval_index, "r", encoding="utf-8") as f:
        eval_files = [json.loads(line) for line in f]

    if len(eval_files) == 0:
        train_files, eval_files = split_data(train_files)

    train_batch_nums = int(len(train_files) / args.batch_size)
    eval_batch_nums = int(len(eval_files) / args.batch_size)
    random.shuffle(train_files)
    random.shuffle(eval_files)
    train_files = train_files[: train_batch_nums * args.batch_size]
    eval_files = eval_files[: eval_batch_nums * args.batch_size]

    strip_prefixes = [s.strip() for s in str(args.strip_prefix).split(",") if s.strip()]
    train_set = NotaGenDataset(
        train_files,
        patchilizer,
        data_root=args.data_root,
        relative_base=args.relative_base,
        strip_prefixes=strip_prefixes,
    )
    eval_set = NotaGenDataset(
        eval_files,
        patchilizer,
        data_root=args.data_root,
        relative_base=args.relative_base,
        strip_prefixes=strip_prefixes,
    )

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank) if world_size > 1 else None
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=local_rank) if world_size > 1 else None

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_batch(b, device),
        sampler=train_sampler,
        shuffle=(train_sampler is None),
    )
    eval_loader = DataLoader(
        eval_set,
        batch_size=args.batch_size,
        collate_fn=lambda b: collate_batch(b, device),
        sampler=eval_sampler,
        shuffle=False,
    )

    if WANDB_LOGGING and global_rank == 0:
        import wandb

        wandb.login(key=WANDB_KEY)
        wandb.init(project="notagen", name=WANDB_NAME)

    best_eval = float("inf")
    best_epoch = 0
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if eval_sampler is not None:
            eval_sampler.set_epoch(epoch)

        if global_rank == 0:
            print("-" * 21 + f"Epoch {epoch}" + "-" * 21)

        unwrap(model).train()
        tqdm_train = tqdm(train_loader, desc=f"train epoch {epoch}", disable=(global_rank != 0))
        running = 0.0
        it = 0

        for batch in tqdm_train:
            input_patches, input_masks = batch
            minibatches = split_into_minibatches(
                input_patches, input_masks, max(1, args.batch_size // max(1, args.accumulation_steps))
            )

            optimizer.zero_grad(set_to_none=True)

            for mb_patches, mb_masks in minibatches:
                with autocast(enabled=scaler.is_enabled()):
                    loss = (
                        unwrap(model)(input_ids=mb_patches, attention_mask=mb_masks).loss
                        / max(1, args.accumulation_steps)
                    )
                scaler.scale(loss).backward()
                running += float(loss.item())

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            global_step += 1
            it += 1
            if global_rank == 0:
                tqdm_train.set_postfix({"loss": running / max(it, 1), "lr": optimizer.param_groups[0]["lr"]})

            if WANDB_LOGGING and global_rank == 0:
                import wandb

                wandb.log({"train_loss": running / max(it, 1)}, step=global_step)

            if global_step % 1000 == 0:
                clear_unused_tensors(model, optimizer)

        eval_loss = evaluate(unwrap(model), eval_loader, world_size, device, scaler.is_enabled())

        if global_rank == 0:
            with open(LOGS_PATH, "a", encoding="utf-8") as f:
                f.write(
                    f"Epoch {epoch}\ntrain_loss: {running / max(it, 1)}\n"
                    f"eval_loss: {eval_loss}\ntime: {time.asctime(time.localtime(time.time()))}\n\n"
                )

            # 保存 LoRA adapter（只保存 adapter 权重与配置，不保存 base 权重）
            epoch_dir = os.path.join(args.output_dir, f"epoch_{epoch}")
            os.makedirs(epoch_dir, exist_ok=True)
            unwrap(model).save_pretrained(epoch_dir)

            # 每个 epoch 都导出一次“合并 LoRA 后”的完整权重（.pth）
            from peft import PeftModel

            m = unwrap(model)
            merged = deepcopy(m).merge_and_unload() if isinstance(m, PeftModel) else deepcopy(m)
            merged.eval()
            merged_ckpt_path = os.path.join(args.ckpt_dir, f"merged_epoch_{epoch}.pth")
            torch.save({"model": merged.state_dict(), "epoch": epoch, "eval_loss": float(eval_loss)}, merged_ckpt_path)

            if eval_loss < best_eval:
                best_eval = eval_loss
                best_epoch = epoch
                best_dir = os.path.join(args.output_dir, "best")
                os.makedirs(best_dir, exist_ok=True)
                unwrap(model).save_pretrained(best_dir)

                merged_best = deepcopy(m).merge_and_unload() if isinstance(m, PeftModel) else deepcopy(m)
                merged_best.eval()
                best_ckpt_path = os.path.join(args.ckpt_dir, "merged_best.pth")
                torch.save({"model": merged_best.state_dict(), "epoch": epoch, "eval_loss": float(eval_loss)}, best_ckpt_path)

            print(f"Epoch {epoch} eval_loss={eval_loss:.6f} best={best_eval:.6f}")

        if world_size > 1:
            dist.barrier()

    # 训练结束：再导出一个 merged_last.pth（最后一次参数）
    if global_rank == 0:
        from peft import PeftModel

        m = unwrap(model)
        merged_last = deepcopy(m).merge_and_unload() if isinstance(m, PeftModel) else deepcopy(m)
        merged_last.eval()
        last_ckpt_path = os.path.join(args.ckpt_dir, "merged_last.pth")
        torch.save({"model": merged_last.state_dict(), "epoch": args.epochs, "best_epoch": best_epoch, "best_eval_loss": float(best_eval)}, last_ckpt_path)
        print(f"Saved merged ckpt: {last_ckpt_path}")


if __name__ == "__main__":
    main()

