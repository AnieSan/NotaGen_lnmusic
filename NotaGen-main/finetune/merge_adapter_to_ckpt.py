import os
import argparse
import torch
from transformers import GPT2Config

from utils import NotaGenLMHeadModel
from config import (
    PATCH_SIZE,
    PATCH_LENGTH,
    PATCH_NUM_LAYERS,
    CHAR_NUM_LAYERS,
    HIDDEN_SIZE,
)


def build_base_model(device):
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
    model.eval()
    return model


def torch_load_weights(path: str, device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def load_base_ckpt(model, ckpt_path: str, device):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"base_ckpt 不存在：{ckpt_path}")
    ckpt = torch_load_weights(ckpt_path, device=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=False)
    return model


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="把 LoRA adapter 合并为完整 ckpt(.pth)")
    ap.add_argument("--base_ckpt", type=str, required=True, help="base 权重 .pth（预训练或全量微调 ckpt）")
    ap.add_argument("--adapter_dir", type=str, required=True, help="LoRA adapter 目录（如 ./lora_adapters/best）")
    ap.add_argument("--out_ckpt", type=str, required=True, help="输出 ckpt 路径（.pth）")
    ap.add_argument("--device", type=str, default="cuda", help="cuda / cpu")
    args = ap.parse_args()

    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")

    model = build_base_model(device)
    model = load_base_ckpt(model, args.base_ckpt, device)

    from peft import PeftModel

    peft_model = PeftModel.from_pretrained(model, args.adapter_dir, is_trainable=False)
    merged = peft_model.merge_and_unload()
    merged.eval()

    os.makedirs(os.path.dirname(args.out_ckpt) or ".", exist_ok=True)
    torch.save({"model": merged.state_dict()}, args.out_ckpt)
    print(f"已导出合并 ckpt: {args.out_ckpt}")


if __name__ == "__main__":
    main()

