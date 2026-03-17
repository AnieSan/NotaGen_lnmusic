import torch
from transformers import GPT2Config
from config import *
from utils import *
from music21 import converter

# =====================================================
# 1️⃣ 修改这里：你的权重路径
# =====================================================
CKPT_PATH = "../../weights_notagen_pretrain_p_size_16_p_length_2048_p_layers_12_c_layers_3_h_size_768_lr_0.0002_batch_8.pth"

# =====================================================
# 2️⃣ 生成参数
# =====================================================
TOP_K = 0
TOP_P = 0.9
TEMPERATURE = 0.8
GENERATE_PATCHES = 40
OUTPUT_FILE = "generated.abc"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

# =====================================================
# 3️⃣ 初始化 Patchilizer
# =====================================================
patchilizer = Patchilizer()

# =====================================================
# 4️⃣ 构建模型（必须和训练时一致）
# =====================================================
patch_config = GPT2Config(
    num_hidden_layers=PATCH_NUM_LAYERS,
    max_length=PATCH_LENGTH,
    max_position_embeddings=PATCH_LENGTH,
    n_embd=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=1
)

char_config = GPT2Config(
    num_hidden_layers=CHAR_NUM_LAYERS,
    max_length=PATCH_SIZE + 1,
    max_position_embeddings=PATCH_SIZE + 1,
    hidden_size=HIDDEN_SIZE,
    num_attention_heads=HIDDEN_SIZE // 64,
    vocab_size=128
)

model = NotaGenLMHeadModel(
    encoder_config=patch_config,
    decoder_config=char_config
).to(device)

# =====================================================
# 5️⃣ 加载权重
# =====================================================
print("Loading checkpoint...")
checkpoint = torch.load(CKPT_PATH, map_location=device)

if "model" in checkpoint:
    model.load_state_dict(checkpoint["model"])
else:
    model.load_state_dict(checkpoint)

model.eval()
print("Model loaded successfully.")

# =====================================================
# 6️⃣ Prompt（必须包含 [V:1]）
# =====================================================
prompt = """X:1
T:Lingnan
M:4/4
L:1/8
K:G
[V:1]
G A B c |
"""

# =====================================================
# 7️⃣ 编码 prompt
# =====================================================
input_patches = patchilizer.encode_generate(prompt)

# ---- 强制 pad 到 PATCH_SIZE ----
fixed_patches = []
for p in input_patches:
    if len(p) < PATCH_SIZE:
        p = p + [0] * (PATCH_SIZE - len(p))
    fixed_patches.append(p)

input_patches = fixed_patches

# =====================================================
# 8️⃣ 初始化生成
# =====================================================
generated_patches = input_patches.copy()

input_tensor = torch.tensor(
    generated_patches,
    dtype=torch.long
).unsqueeze(0).to(device)

print("Start generating patches...")

# =====================================================
# 9️⃣ Patch-level 自回归生成
# =====================================================
for step in range(GENERATE_PATCHES):

    with torch.no_grad():
        new_patch = model.generate(
            input_tensor,
            top_k=TOP_K,
            top_p=TOP_P,
            temperature=TEMPERATURE
        )

    # 如果返回 tensor，转 list
    if isinstance(new_patch, torch.Tensor):
        new_patch = new_patch.squeeze().tolist()

    # 强制长度为 PATCH_SIZE
    if len(new_patch) < PATCH_SIZE:
        new_patch += [0] * (PATCH_SIZE - len(new_patch))

    generated_patches.append(new_patch)

    input_tensor = torch.tensor(
        generated_patches,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    print(f"Generated patch {step+1}/{GENERATE_PATCHES}")

# =====================================================
# 🔟 解码为 ABC
# =====================================================
abc_text = patchilizer.decode(generated_patches)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(abc_text)

print("ABC saved to:", OUTPUT_FILE)

# =====================================================
# 1️⃣1️⃣ 转 MIDI
# =====================================================
try:
    score = converter.parseData(abc_text, format="abc")
    midi_path = OUTPUT_FILE.replace(".abc", ".mid")
    score.write("midi", midi_path)
    print("MIDI exported to:", midi_path)
except Exception as e:
    print("MIDI export failed:", e)