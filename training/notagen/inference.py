import os
import time
import torch
from utils import *
from config import *
from transformers import GPT2Config, LlamaConfig
from abctoolkit.utils import extract_metadata_and_tunebody_rotated


def time_unreduce(abc_text):
    # unreduce
    abc_lines = abc_text.split('\n')
    
    tunebody_index = None
    for i in range(len(abc_lines)):
        if abc_lines[i].startswith('[r:'):
            tunebody_index = i
            break

    metadata_lines = abc_lines[ : tunebody_index]
    tunebody_lines = abc_lines[tunebody_index : ]

    part_symbol_list = []
    for line in metadata_lines:
        if line.startswith('V:'):
            part_symbol_list.append(line.split()[0])
    # part_symbol_list = sorted(part_symbol_list)

    last_visible_bar = {}
    for symbol in part_symbol_list:
        last_visible_bar[symbol] = None

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''
        match = re.match(r'^\[r:\d+\]', line)
        if match:
            line_annotation = match.group(0)
            unreduced_line += line_annotation

        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)
        
        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            line_bar_dict[key] = value

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                symbol_bartext = line_bar_dict[symbol]
                last_visible_bar[symbol] = symbol_bartext
            else:
                symbol_bartext = last_visible_bar[symbol]
            unreduced_line += '[' + symbol + ']' + symbol_bartext
        
        unreduced_tunebody_lines.append(unreduced_line)

    unreduced_lines = metadata_lines + unreduced_tunebody_lines
    unreduced_abc_text = '\n'.join(unreduced_lines)

    return unreduced_abc_text



if torch.cuda.is_available():    
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

patchilizer = Patchilizer()

if PATCH_DECODER_STRUCTURE == 'gpt2':
    patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                        max_length=PATCH_LENGTH, 
                        max_position_embeddings=PATCH_LENGTH,
                        n_embd=HIDDEN_SIZE,
                        num_attention_heads=HIDDEN_SIZE//64,
                        vocab_size=1)
elif PATCH_DECODER_STRUCTURE == 'llama':
    patch_config = LlamaConfig(num_hidden_layers=PATCH_NUM_LAYERS,
                               max_length=PATCH_LENGTH, 
                               max_position_embeddings=PATCH_LENGTH,
                               hidden_size=HIDDEN_SIZE,
                               num_attention_heads=HIDDEN_SIZE//64, 
                               intermediate_size=HIDDEN_SIZE*4,
                               vocab_size=1)
if BYTE_DECODER_STRUCTURE == 'gpt2':
    byte_config = GPT2Config(num_hidden_layers=BYTE_NUM_LAYERS, 
                             max_length=PATCH_SIZE+1, 
                             max_position_embeddings=PATCH_SIZE+1,
                             hidden_size=HIDDEN_SIZE,
                             num_attention_heads=HIDDEN_SIZE//64,
                             vocab_size=128)
elif BYTE_DECODER_STRUCTURE == 'llama':
    byte_config = LlamaConfig(num_hidden_layers=BYTE_NUM_LAYERS,
                              max_length=PATCH_SIZE+1, 
                              max_position_embeddings=PATCH_SIZE+1,
                              hidden_size=HIDDEN_SIZE,
                              num_attention_heads=HIDDEN_SIZE//64,
                              intermediate_size=HIDDEN_SIZE*4,
                              vocab_size=128)
    
model = bGPTLMHeadModel(encoder_structure=PATCH_DECODER_STRUCTURE, decoder_structure=BYTE_DECODER_STRUCTURE,
                        encoder_config=patch_config, decoder_config=byte_config)

print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()

# bos_patch = [byte for byte in bytearray(TARGET_EXT, 'utf-8')]
# bos_patch = bos_patch + [256] * (PATCH_SIZE - len(bos_patch))
bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]
eos_patch = [patchilizer.bos_token_id] + [patchilizer.eos_token_id] * (PATCH_SIZE - 1)

files = list(range(NUM_SAMPLES))

for i in files:
    filename = OUTPUT_FOLDER+"/"+time.strftime("%Y%m%d-%H%M%S")+"-"+str(i+1)+"."+TARGET_EXT
    byte_list = bos_patch.copy()
    prefix_len = len(byte_list)
    input_patches = torch.tensor([byte_list], device=device)

    end_flag = False
    cut_index = None
    while True:
        predicted_patch = model.generate(input_patches.unsqueeze(0),
                                         top_k=TOP_K,
                                         top_p=TOP_P,
                                         temperature=TEMPERATURE)
        if predicted_patch == eos_patch:
            end_flag = True
            break
        # print(predicted_patch)
        for byte in predicted_patch:
            if byte == 0:
                break
            char = chr(byte)
            byte_list.append(char)

        # byte_list.append(patchilizer.decode(predicted_patch))
        predicted_patch = torch.tensor([predicted_patch], device=device)    # (1, 16)
        input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

        if len(byte_list) > 102400:
            break

        if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag :
            # 做流式切片
            abc_code = ''.join(byte_list[prefix_len:])
            abc_lines = abc_code.split('\n')

            tunebody_index = None
            for i, line in enumerate(abc_lines):
                if line.startswith('[r:'):
                    tunebody_index = i
                    break
            if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                # 生成的全是metadata，放弃
                break

            metadata_lines = abc_lines[:tunebody_index]
            tunebody_lines = abc_lines[tunebody_index:]

            if cut_index is None:
                cut_index = len(tunebody_lines) // 2
            abc_code_slice = '\n'.join(metadata_lines + tunebody_lines[-cut_index : ])
            input_patches = patchilizer.encode(abc_code_slice, generate_mode=True)
            input_patches = [item for sublist in input_patches for item in sublist]
            # input_patches = bos_patch + input_patches

            input_patches = torch.tensor([input_patches], device=device)
            input_patches = input_patches.reshape(1, -1)
            # print(byte_list)

    byte_list = byte_list[prefix_len:]
    abc_text = ''.join(byte_list)

    # try:
    #     abc_text = time_unreduce(abc_text)
    # except:
    #     continue

    # set output file name as the current time
    with open(filename, 'w') as file:
        file.write(abc_text)
        print("Generated "+filename)
