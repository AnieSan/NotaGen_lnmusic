import os
import time
import torch
from utils import *
from config import *
from transformers import GPT2Config, LlamaConfig
from abctoolkit.utils import Exclaim_re, Quote_re, SquareBracket_re
from abctoolkit.transpose import Note_list, Pitch_sign_list

Note_list = Note_list + ['z', 'x']

def voice_unreduce(abc_lines):

    tunebody_index = None
    for i in range(len(abc_lines)):
        if abc_lines[i].startswith('[r:') or abc_lines[i].startswith('[V:'):
            tunebody_index = i
            break

    metadata_lines = abc_lines[ : tunebody_index]
    tunebody_lines = abc_lines[tunebody_index : ]

    part_symbol_list = []
    for line in metadata_lines:
        if line.startswith('V:'):
            part_symbol_list.append(line.split()[0])

    part_symbol_list.sort(key=lambda x: int(x[2:]))

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''
        match = re.match(r'\[r:([^\]]+)\]', line)
        if match:
            line_annotation = match.group(0)
            unreduced_line += line_annotation

        pattern = r'\[V:([^\]]+)\](.*?)(?=\[V:|$)'
        matches = re.findall(pattern, line)

        line_bar_group_dict = {}
        line_bar_dict = {}
        for match in matches:
            key = f'V:{match[0]}'
            value = match[1]
            symbol_index_group = key[2:]
            line_bar_group_dict[symbol_index_group] = value

        for symbol_index_group, bar_group_text in line_bar_group_dict.items():
            same_index_group_list = symbol_index_group.split(';')
            group_notes_text_list = []
            group_notes_start_list = []
            group_notes_end_list = []

            start_char = re.escape(chr(21))
            end_char = re.escape(chr(22))
            group_note_pattern = re.compile(f'{start_char}.*?{end_char}')

            group_note_matches = group_note_pattern.finditer(bar_group_text)
            for group_match in group_note_matches:
                group_start = group_match.start()
                group_end = group_match.end()
                group_text = group_match.group()

                group_notes_text_list.append(group_text)
                group_notes_start_list.append(group_start)
                group_notes_end_list.append(group_end)

            for j, same_index_group in enumerate(same_index_group_list):
                # 复原 bar text
                bar_text = ''
                if len(group_notes_text_list) > 0:
                    bar_text += bar_group_text[ : group_notes_start_list[0]]
                    for k, group_notes_text in enumerate(group_notes_text_list):
                        specific_note = group_notes_text[1:-1].split(';')[j]
                        bar_text += specific_note
                        if k < len(group_notes_text_list) - 1:
                            bar_text += bar_group_text[group_notes_end_list[k] : group_notes_start_list[k+1]]
                        else:
                            bar_text += bar_group_text[group_notes_end_list[k] : ]
                else:
                    bar_text = bar_group_text

                same_index_list = same_index_group.split(',')
                for index in same_index_list:
                    symbol = 'V:' + index
                    line_bar_dict[symbol] = bar_text

        for symbol in part_symbol_list:
            if symbol in line_bar_dict.keys():
                unreduced_line = unreduced_line + '[' + symbol + ']'
                unreduced_line += line_bar_dict[symbol]

        unreduced_tunebody_lines.append(unreduced_line)

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


def time_unreduce(abc_lines):
    
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

    last_visible_bar = {}
    for symbol in part_symbol_list:
        last_visible_bar[symbol] = None

    unreduced_tunebody_lines = []

    for i, line in enumerate(tunebody_lines):
        unreduced_line = ''
        match = re.match(r'\[r:([^\]]+)\]', line)
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
            if symbol_bartext is not None:  # 加上此条件，能避免一些漏声部的情况
                unreduced_line += '[' + symbol + ']' + symbol_bartext
        
        unreduced_tunebody_lines.append(unreduced_line)

    unreduced_lines = metadata_lines + unreduced_tunebody_lines

    return unreduced_lines


def format_a_bar_text(bar_text):
    index_list = [-1] * len(bar_text)   # -1：未定，0：非音符，1：音符(A/z)

    # 屏蔽[metadata], !!, ""
    squareBracket_matches = re.finditer(SquareBracket_re, bar_text)
    for squareBracket_match in squareBracket_matches:
        sqaureBracket_start = squareBracket_match.start()
        sqaureBracket_end = squareBracket_match.end()
        squareBracket_string = squareBracket_match.group()
        if squareBracket_string[2] == ':':  # metadata
            for j in range(sqaureBracket_start, sqaureBracket_end):
                index_list[j] = 0

    exclaim_matches = re.finditer(Exclaim_re, bar_text)
    for exclaim_match in exclaim_matches:
        exclaim_start = exclaim_match.start()
        exclaim_end = exclaim_match.end()
        for j in range(exclaim_start, exclaim_end):
            index_list[j] = 0

    quote_matches = re.finditer(Quote_re, bar_text)
    for quote_match in quote_matches:
        quote_start = quote_match.start()
        quote_end = quote_match.end()
        for j in range(quote_start, quote_end):
            index_list[j] = 0

    # 屏蔽非音符字符
    for i, char in enumerate(bar_text):
        if index_list[i] == -1:
            if char not in Note_list + Pitch_sign_list:
                index_list[i] = 0
            else:
                index_list[i] = 1

    # 挑出音符和休止符
    note_re = r'[=^_]*[A-Ga-g](?:\'*,*)'
    rest_re = r'[zx]'

    note_rest_list = []

    i = 0
    note_section_string_list = []
    while i < len(index_list):
        if index_list[i] == 1:
            j = i + 1
            while j < len(index_list):
                if index_list[j] == 1:
                    j += 1
                else:
                    break
            note_section_string = bar_text[i:j]
            note_section_string_list.append(note_section_string)

            note_matches = re.finditer(note_re, note_section_string)
            for note_match in note_matches:
                note_start = note_match.start() + i
                note_end = note_match.end() + i
                note_string = note_match.group()
                note_rest_list.append({'start': note_start, 'end': note_end, 'content': note_string})

            rest_matches = re.finditer(rest_re, note_section_string)
            for rest_match in rest_matches:
                rest_start = rest_match.start() + i
                rest_end = rest_match.end() + i
                rest_string = rest_match.group()
                note_rest_list.append({'start': rest_start, 'end': rest_end, 'content': rest_string})

            i = j + 1
        else:
            i += 1

    note_rest_list.sort(key=lambda x: x['start'])

    bar_text_formatted = bar_text
    # 替换音符和休止符：音符->chr(11)，休止符->chr(12)
    for ele_dict in reversed(note_rest_list):
        if ele_dict['content'] not in ['z', 'x']:
            bar_text_formatted = bar_text_formatted[0:ele_dict['start']] + chr(11) + bar_text_formatted[ele_dict['end']:]
        else:
            bar_text_formatted = bar_text_formatted[0:ele_dict['start']] + chr(12) + bar_text_formatted[ele_dict['end']:]

    assert bar_text_formatted.count(chr(11)) + bar_text_formatted.count(chr(12)) == len(note_rest_list)

    return bar_text_formatted, note_rest_list



def voice_reduce_again(abc_lines):

    tunebody_index = None
    for i, line in enumerate(abc_lines):
        if line.startswith('[V:') :
            tunebody_index = i
            break

    metadata_lines = abc_lines[:tunebody_index]
    tunebody_lines = abc_lines[tunebody_index:]

    reduced_abc_lines = metadata_lines

    for i in range(len(tunebody_lines)):
        reduced_abc_line = ''

        # 提取开头[r:]
        line_annotation_match = re.match(r'\[r:([^\]]+)\]', line)
        if line_annotation_match:
            line_annotation = line_annotation_match.group(0)
            reduced_abc_line += line_annotation

        line = tunebody_lines[i][len(line_annotation):]

        # 格式化
        # 提取小节文本
        bar_formatted_dict = {}
        pattern = r'\[V:(\d+)\](.*?)(?=\[V:|$)'
    
        matches = re.findall(pattern, line, re.DOTALL)

        for match in matches:
            symbol = f'V:{match[0]}'
            bar = match[1]
            bar_formatted, note_rest_list = format_a_bar_text(bar)
            bar_formatted_dict[symbol] = {'bar': bar, 
                                          'bar_formatted': bar_formatted, 
                                          'note_rest_list': note_rest_list}

        line_bar_formatted_dict = {}
        for symbol in bar_formatted_dict.keys():
            bar_formatted = bar_formatted_dict[symbol]['bar_formatted']
            if bar_formatted not in line_bar_formatted_dict.keys():
                line_bar_formatted_dict[bar_formatted] = [symbol]
            else:
                line_bar_formatted_dict[bar_formatted].append(symbol)
            pass

        # 相同格式做合并
        for bar_format, symbol_list in line_bar_formatted_dict.items():
            merged_bar_text = ''

            # 挑出小节文本完全相同的组
            same_text_bar_dict = {}
            for symbol in symbol_list:
                bar = bar_formatted_dict[symbol]['bar']
                if bar not in same_text_bar_dict.keys():
                    same_text_bar_dict[bar] = [int(symbol[2:])] # 这里只插入数字（方便排序）
                else:
                    same_text_bar_dict[bar].append(int(symbol[2:]))
            same_text_index_group_list = []
            for same_text_index_list in same_text_bar_dict.values():
                same_text_index_list.sort()
                same_text_index_group_list.append(same_text_index_list)
            same_text_index_group_list.sort(key=lambda x: x[0])

            symbol_string = [','.join(list(map(str, group))) for group in same_text_index_group_list]
            symbol_string = ';'.join(symbol_string)
            merged_bar_text = '[V:' + symbol_string + ']'

            j = 0
            note_rest_index = 0 # note_rest_list轮到第几个元素
            while j < len(bar_format):
                if bar_format[j] not in [chr(11), chr(12)]:
                    merged_bar_text += bar_format[j]
                else:
                    note_rest_group = []
                    for same_text_index_group in same_text_index_group_list:
                        repr_symbol = 'V:' + str(same_text_index_group[0])
                        note_rest_group.append(bar_formatted_dict[repr_symbol]['note_rest_list'][note_rest_index]['content'])
                    note_rest_index += 1
                    if len(set(note_rest_group)) > 1:
                        note_rest_group_text = chr(21) + ';'.join(note_rest_group) + chr(22)
                    else:
                        note_rest_group_text = note_rest_group[0]
                    merged_bar_text += note_rest_group_text

                j += 1

            reduced_abc_line += merged_bar_text

        reduced_abc_line += '\n'
        reduced_abc_lines.append(reduced_abc_line)

    return reduced_abc_lines



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
                        vocab_size=50000 if PATCH_MODE == 'bpe' else 1)
elif PATCH_DECODER_STRUCTURE == 'llama':
    patch_config = LlamaConfig(num_hidden_layers=PATCH_NUM_LAYERS,
                               max_length=PATCH_LENGTH,
                               max_position_embeddings=PATCH_LENGTH,
                               hidden_size=768,
                               num_attention_heads=HIDDEN_SIZE//64,
                               intermediate_size=HIDDEN_SIZE*4,
                               vocab_size=50000 if PATCH_MODE == 'bpe' else 1)
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
    
if PATCH_MODE == 'bpe':
    model = bpeLMHeadModel(structure=PATCH_DECODER_STRUCTURE, config=patch_config)
else:
    model = bGPTLMHeadModel(encoder_structure=PATCH_DECODER_STRUCTURE, decoder_structure=BYTE_DECODER_STRUCTURE,
                            encoder_config=patch_config, decoder_config=byte_config)

print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

checkpoint = torch.load(INFERENCE_WEIGHTS_PATH, map_location=torch.device(device))
model.load_state_dict(checkpoint['model'])
model = model.to(device)
model.eval()


files = list(range(NUM_SAMPLES))


def inference_patch():

    bos_patch = [patchilizer.bos_token_id] * (PATCH_SIZE - 1) + [patchilizer.eos_token_id]

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
            if predicted_patch[0] == patchilizer.bos_token_id and predicted_patch[1] == patchilizer.eos_token_id:
                end_flag = True
                break
            next_patch = patchilizer.decode([predicted_patch])

            for char in next_patch:
                byte_list.append(char)
                print(char, end='')

            patch_end_flag = False
            for j in range(len(predicted_patch)):
                if patch_end_flag:
                    predicted_patch[j] = patchilizer.special_token_id
                if predicted_patch[j] == patchilizer.eos_token_id:
                    patch_end_flag = True

            predicted_patch = torch.tensor([predicted_patch], device=device)    # (1, 16)
            input_patches = torch.cat([input_patches, predicted_patch], dim=1)  # (1, 16 * patch_len)

            if len(byte_list) > 102400:
                break

            if input_patches.shape[1] >= PATCH_LENGTH * PATCH_SIZE and not end_flag :
                # 做流式切片
                print('Stream generating...')
                abc_code = ''.join(byte_list[prefix_len:])
                abc_lines = abc_code.split('\n')

                tunebody_index = None
                for i, line in enumerate(abc_lines):
                    if line.startswith('[r:') or line.startswith('[V:'):
                        tunebody_index = i
                        break
                if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                    # 生成的全是metadata，放弃
                    break

                metadata_lines = abc_lines[:tunebody_index]
                tunebody_lines = abc_lines[tunebody_index:]

                metadata_lines = [line + '\n' for line in metadata_lines]
                if not abc_code.endswith('\n'): # 如果生成结果最后一行未完结
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
                else:
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                if cut_index is None:
                    cut_index = len(tunebody_lines) // 2

                if REDUCE_TYPE in ['time', 'time-voice']:
                    # 还原 -cut_index 行的内容
                    lines_to_unreduce = metadata_lines + tunebody_lines[ : -cut_index + 1]
                    try:
                        if REDUCE_TYPE == 'time':
                            unreduced_lines = time_unreduce(lines_to_unreduce)
                        elif REDUCE_TYPE == 'time-voice':
                            unreduced_lines = voice_unreduce(lines_to_unreduce)
                            unreduced_lines = time_unreduce(unreduced_lines)
                            reduce_again_lines = voice_reduce_again(unreduced_lines)
                    except:
                        print('Syntax fault in unreduction')
                        break

                    unreduced_cut_index_line = unreduced_lines[-1]
                    abc_code_slice = ''.join(metadata_lines + [unreduced_cut_index_line] + tunebody_lines[-cut_index + 1 :])
                    input_patches = patchilizer.encode_generate(abc_code_slice)
                else:
                    abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index : ])
                    input_patches = patchilizer.encode_generate(abc_code_slice)

                input_patches = [item for sublist in input_patches for item in sublist]
                input_patches = torch.tensor([input_patches], device=device)
                input_patches = input_patches.reshape(1, -1)
                # print(byte_list)

        byte_list = byte_list[prefix_len:]
        # print(''.join(byte_list))
        # set output file name as the current time

        abc_text = ''.join(byte_list)

        # unreduce

        with open(filename, 'w') as file:
            file.write(''.join(byte_list))
            print("Generated "+filename)


def inference_bpe():

    bos_patch = [patchilizer.bos_token_id]

    for i in files:
        filename = OUTPUT_FOLDER + "/" + time.strftime("%Y%m%d-%H%M%S") + "-" + str(i + 1) + "." + TARGET_EXT
        token_list = []
        prefix_len = len(token_list)
        input_patches = torch.tensor([bos_patch], device=device)
        word_list = []
        end_flag = False
        cut_index = None
        while True:
            generated_ids = model.generate(input_patches,
                                             top_k=TOP_K,
                                             top_p=TOP_P,
                                             temperature=TEMPERATURE)

            for token_id in generated_ids:
                token_list.append(token_id)
                token_word = patchilizer.bpe_tokenizer.id_to_piece(token_id)
                word_list.append(token_word)
                print(token_word, end='')

            if int(generated_ids[-1]) == patchilizer.eos_token_id:
                end_flag = True
                break

            input_patches = torch.cat((input_patches, torch.tensor([generated_ids], device=device)), dim=-1)   # [1, seq_len]

            if len(word_list) > 102400:
                break

            if input_patches.shape[1] >= PATCH_LENGTH and not end_flag:
                # 做流式切片
                print('Stream generating...')

                abc_code = patchilizer.bpe_tokenizer.decode_ids(token_list[prefix_len:])
                # print('abccode\n', abc_code)
                abc_lines = abc_code.split('\n')
                abc_lines = [line.lstrip() for line in abc_lines]   # 去掉某些行可能开头有的空格

                tunebody_index = None
                for i, line in enumerate(abc_lines):
                    if line.startswith('[r:') or line.startswith('[V:'):
                        tunebody_index = i
                        break
                if tunebody_index is None or tunebody_index == len(abc_lines) - 1:
                    # 生成的全是metadata，放弃
                    break

                metadata_lines = abc_lines[:tunebody_index]
                tunebody_lines = abc_lines[tunebody_index:]

                metadata_lines = [line + '\n' for line in metadata_lines]
                if not abc_code.endswith('\n'):  # 如果生成结果最后一行未完结
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [
                        tunebody_lines[-1]]
                else:
                    tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]

                if cut_index is None:
                    cut_index = len(tunebody_lines) // 2

                if REDUCE_TYPE in ['time', 'time-voice']:
                    # 还原 -cut_index 行的内容
                    lines_to_unreduce = metadata_lines + tunebody_lines[: -cut_index + 1]
                    try:
                        if REDUCE_TYPE == 'time':
                            unreduced_lines = time_unreduce(lines_to_unreduce)
                        elif REDUCE_TYPE == 'time-voice':
                            unreduced_lines = voice_unreduce(lines_to_unreduce)
                            unreduced_lines = time_unreduce(unreduced_lines)
                            reduce_again_lines = voice_reduce_again(unreduced_lines)
                    except:
                        print('Syntax fault in unreduction')
                        break

                    unreduced_cut_index_line = unreduced_lines[-1]
                    abc_code_slice = ''.join(
                        metadata_lines + [unreduced_cut_index_line] + tunebody_lines[-cut_index + 1:])
                    input_patches = patchilizer.encode_generate(abc_code_slice)
                else:
                    abc_code_slice = ''.join(metadata_lines + tunebody_lines[-cut_index:])
                    input_patches = patchilizer.encode_generate(abc_code_slice)

                input_patches = torch.tensor([input_patches], device=device)
                input_patches = input_patches.reshape(1, -1)

        word_list = word_list[prefix_len:]

        abc_text = patchilizer.bpe_tokenizer.decode_ids(token_list[prefix_len:])

        # unreduce

        with open(filename, 'w') as file:
            file.write(abc_text)
            print("Generated " + filename)


if __name__ == '__main__':

    if PATCH_MODE in ['bar', 'barbyte', 'linebyte', 'byte']:
        inference_patch()
    elif PATCH_MODE == 'bpe':
        inference_bpe()