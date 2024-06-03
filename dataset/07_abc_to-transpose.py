import os
import re
from unidecode import unidecode
from fractions import Fraction
from utils import find_all_abc
from multiprocessing import Pool
from tqdm import tqdm

ORI_DIR = r'D:\Research\Projects\MultitrackComposer\dataset\08_abc_key-augmented\musescoreV2'
DES_DIR = r'D:\Research\Projects\MultitrackComposer\dataset\09_abc_transposed\musescoreV2'



def calculate_single_note_duration(note_text):
    note_text = note_text.strip()
    dur_text = re.sub(r'[a-zA-Z]', '', note_text)
    if dur_text == '':
        dur = Fraction('1')
    elif dur_text == '/':
        dur = Fraction('1/2')
    elif dur_text.startswith('/'):
        dur = Fraction('1' + dur_text)
    else:
        dur = Fraction(dur_text)
    return dur


def calculate_duration_V2(bar_text, abc_path):
    # 尝试仅使用文本方法计算小节时值
    # print(bar_text)
    original_bartext = bar_text

    note_set = ['A', 'a', 'B', 'b', 'C', 'c', 'D', 'd', 'E', 'e', 'F', 'f', 'G', 'g', 'X', 'x', 'Z', 'z']
    useless_char_set = ['P', 'S', 'O', 's', 'o', 'u', 'v', 'U', 'V', 'T', 'M', '.', '-', ')']
    pitch_sign_set = ['_', '=', '^', '\'', ',']
    time_sign_set = ['/', ]

    # 去掉小节线
    barlines = ["|:", "::", ":|", "[|", "||", "|]", "|"]
    barlines_regexPattern = '(' + '|'.join(map(re.escape, barlines)) + ')'
    bar_text = re.split(barlines_regexPattern, bar_text)[0]

    # 删掉 !! "" {} 内容
    exclaim_re = r'![^!]+!'
    quote_re = r'"[^"]+"'
    brace_re = r'\{[^}]+\}'
    bar_text = re.sub(quote_re, '', bar_text)
    bar_text = re.sub(exclaim_re, '', bar_text)
    bar_text = re.sub(brace_re, '', bar_text)

    # 处理[]内容
    bracket_re = r'\[[^\]]+\]'
    bracket_contents = re.findall(bracket_re, bar_text)
    # 如果里面是元信息切换，则删掉
    for bracket_content in bracket_contents:
        if len(bracket_content) >= 2 and (bracket_content[1] not in note_set) and bracket_content[2] == ':':
            bar_text = bar_text.replace(bracket_content, '')
    # # 如果是和弦，则 打印
    bracket_contents = re.findall(bracket_re, bar_text)
    if len(bracket_contents) > 0:
        for bracket_content in bracket_contents:
            # 经检查，[]内容不会超出以下范围: note_set +  pitch_sign_set + '[' + ']' + '-'
            # 故可将[]内容替换为 C
            for char in bracket_content:
                if char not in note_set + pitch_sign_set + ['[', ']', '-']:
                    print(bracket_content)
            bar_text = bar_text.replace(bracket_content, 'C')

    # 将x,z换成A
    bar_text = bar_text.replace('x', 'z')
    bar_text = bar_text.replace('X', 'z')

    # 替换无用字符和音高表示字符
    for char in useless_char_set + pitch_sign_set:
        bar_text = bar_text.replace(char, '')
    # 处理(，如果(后面跟了数字，则留下，否则删掉
    index_to_detele = []
    for i, char in enumerate(bar_text):
        if bar_text[i] == '(' and not bar_text[i+1].isnumeric():
            index_to_detele.append(i)
    bar_text_list = list(bar_text)
    # print(bar_text_list, index_to_detele)
    for index in reversed(index_to_detele):
        bar_text_list.pop(index)

    bar_text = ''.join(bar_text_list)

    note_list = []
    # 处理各个字符，整理进入note_list
    index = 0
    for i, char in enumerate(bar_text):
        if i < index:
            continue
        # 遇到音名，往后试探，直到遇到:音名, >, <, (
        if char in note_set:
            index = i + 1
            while index < len(bar_text):
                if bar_text[index] in note_set:
                    break
                elif bar_text[index] == '(':
                    break
                elif bar_text[index] in ['<', '>']: # 附点音符，再往后探到下一个音名
                    index += 1
                index += 1
            note_list.append(bar_text[i:index])
        elif char == '(': # 三连音之类的
            # 判断时值组合方式（截到最后一个数字和冒号）
            num_index = i + 1
            while bar_text[num_index].isnumeric() or bar_text[num_index] == ':':
                num_index += 1
            # 经检查，好像只有(3会简写，其他都会写成(p:q:n的标准格式，肥肠好
            grouping_text = bar_text[i:num_index]
            if grouping_text == '(3':
                # 往后找三个音
                grouping_note_count = 3
            else:
                grouping_note_count = int(grouping_text.split(':')[-1])
            note_count = 0
            index = num_index
            while note_count < grouping_note_count:
                if bar_text[index] in note_set:
                    note_count += 1
                    # 向后探到音符结束
                    note_index = index + 1
                    while note_index < len(bar_text):
                        if bar_text[note_index] in note_set:
                            break
                        elif bar_text[note_index] == '(':
                            break
                        note_index += 1
                    index = note_index
                else:
                    index += 1
            note_list.append(bar_text[i:index])

    note_list = [ele.strip() for ele in note_list]
    bar_duration = 0

    try:
        for ele in note_list:
            if '(' in ele:  # 连音
                num_index = 1
                while ele[num_index].isnumeric() or ele[num_index] == ':':
                    num_index += 1
                grouping_text = ele[:num_index]
                p, q, r = None, None, None
                if grouping_text == '(3':
                    p, q, r = 3, 2, 3
                else:
                    p = int(grouping_text.lstrip('(').split(':')[0])
                    q = int(grouping_text.lstrip('(').split(':')[1])
                    r = int(grouping_text.lstrip('(').split(':')[2])
                group_note_list = re.findall(r'[a-zA-Z][^a-zA-Z]*', ele[num_index:])
                if len(group_note_list) != r:
                    raise Exception('Unmatched r')
                dur = 0
                for note in group_note_list:
                    dur += calculate_single_note_duration(note)
                dur = dur * Fraction(numerator=q, denominator=p)

            elif '<' in ele or '>' in ele:  # 附点
                notes_text = ele.replace('<', ' ')
                notes_text = notes_text.replace('>', ' ')
                note_1 = notes_text.split()[0]
                dur = calculate_single_note_duration(note_1) * 2

            else:   # 单纯的音符
                dur = calculate_single_note_duration(ele)

            bar_duration += dur
    except Exception as e:
        print(bar_text, e)



    print(bar_text, bar_duration)
    # print(note_list)
    # print('')


def test_calculate_duration():
    # 转置abc
    reserved_info_field = ['L:', 'K:', 'M:', 'Q:', 'V:']

    for abc_path in find_all_abc('04_abc_cleaned\\pd2'):
        dataset_folder = abc_path.split('\\')[-2]
        abc_name = abc_path.split('\\')[-1].split('.')[0]
        # print(abc_path)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text_lines = f.readlines()

        global_M = ''
        global_L = ''
        # 滤掉除 Q:K:M:L:V: 以外的 information field
        # 滤掉除 %%score 以外的 %%行
        # 240216修改：这一步提前做
        filtered_abc_text_lines = []
        for i, line in enumerate(abc_text_lines):
            save_state = True
            if re.search(r'^[A-Za-z]:', line) and line[:2] not in reserved_info_field:
                save_state = False
            if line.startswith("%") and not line.startswith('%%score'):
                save_state = False
            if line.startswith('M:'):
                global_M = line.strip()
            if line.startswith('L:'):
                global_L = line.strip()
            if save_state:
                filtered_abc_text_lines.append(line)

        if global_M.lstrip('M:') == 'none':
            continue

        # 分割为各个声部
        part_symbol_list = []

        tunebody_index = None
        for i, line in enumerate(filtered_abc_text_lines):
            if line == 'V:1\n':
                tunebody_index = i
                break
        if tunebody_index is None:
            raise Exception('tunebody index not found.')

        tunebody_lines = filtered_abc_text_lines[tunebody_index:]
        metadata_lines = filtered_abc_text_lines[:tunebody_index]
        part_text_list = []

        last_start_index = None
        for i, line in enumerate(tunebody_lines):
            if i == 0:
                last_start_index = 1
                part_symbol_list.append(line[:3])
                continue
            if line.startswith('V:'):
                last_end_index = i
                part_text_list.append(''.join(tunebody_lines[last_start_index:last_end_index]))
                part_symbol_list.append(line[:3])
                last_start_index = i + 1
        part_text_list.append(''.join(tunebody_lines[last_start_index:]))

        # 240206添加：通过bar patch检查每个声部的小节数能否对得上
        bar_equality_flag = True
        for i in range(1, len(part_text_list)):
            if not check_bar_patch_num_equality(part_text_list[0], part_text_list[i]):
                print('Warning: unequal bar number ', i)
                bar_equality_flag = False
                break
        if not bar_equality_flag:
            continue

        part_patches_list = []
        for i in range(len(part_text_list)):
            part_patches = transposer.split_bars(part_text_list[i])
            for part_patch in part_patches:
                calculate_duration_V2(part_patch, abc_path)


class Transposer:
    """
    A class for transposing abc
    """

    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'

    def split_bars(self, body):
        """
        Split a body of music into individual bars.
        """
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        for j in range(len(bars)):
            # strip，去掉\n，去掉$
            bars[j] = bars[j].strip().replace('\n', '').replace('$', '')
            # 如果以数字开头，则提取数字之后的字符串，直到非数字/,/./-出现，把它加到上一个patch末尾
            if re.match(r'\d', bars[j]):
                k = 0
                for k in range(len(bars[j])):
                    if not bars[j][k] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', ',', '.', '-']:
                        break
                affix = bars[j][:k]
                bars[j] = bars[j][k:].strip()
                bars[j - 1] = bars[j - 1] + affix

        return bars


transposer = Transposer()


def transposed_an_abc_file(abc_path: str, addR=False):
    abc_name = abc_path.split('\\')[-1][:-4]
    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_text_lines = f.readlines()

    # 记录 global_M & global_L
    if addR:
        global_M = ''
        global_L = ''
        for i, line in enumerate(abc_text_lines):
            if line.startswith('M:'):
                global_M = line.strip()
            if line.startswith('L:'):
                global_L = line.strip()

        if global_M.lstrip('M:') == 'none':
            return

    # 分割为各个声部
    part_symbol_list = []

    tunebody_index = None
    for i, line in enumerate(abc_text_lines):
        if line == 'V:1\n':
            tunebody_index = i
            break
    if tunebody_index is None:
        raise Exception('tunebody index not found.')

    tunebody_lines = abc_text_lines[tunebody_index:]
    metadata_lines = abc_text_lines[:tunebody_index]
    part_text_list = []

    last_start_index = None
    for i, line in enumerate(tunebody_lines):
        if i == 0:
            last_start_index = 1
            part_symbol_list.append(line[:3])
            continue
        if line.startswith('V:'):
            last_end_index = i
            part_text_list.append(''.join(tunebody_lines[last_start_index:last_end_index]))
            part_symbol_list.append(line[:3])
            last_start_index = i + 1
    part_text_list.append(''.join(tunebody_lines[last_start_index:]))

    part_patches_list = []
    for i in range(len(part_text_list)):
        part_patches = transposer.split_bars(part_text_list[i])

        if addR:
            # 法一：根据M和L推算，但无法照顾到弱起小节的情况
            # 法二：根据实际小节时值推算，使用music21
            M_value = Fraction(global_M.lstrip('M:'))
            L_value = Fraction(global_L.lstrip('L:'))
            for j in range(len(part_patches)):
                M_matches = re.findall(r'\[M:[^\]]*\]', part_patches[j])
                if len(M_matches) > 1:
                    print(abc_path)
                    raise Exception
                if M_matches:
                    M_value = Fraction(M_matches[0].lstrip('[M:').rstrip(']'))
                try:
                    duration = str(calculate_duration(part_patches[j]))
                except Exception:
                    duration = str(float(M_value / L_value))

                duration = duration.rstrip('.0')
                part_patches[j] = '[r:' + duration + ']' + part_patches[j]

        part_patches = ['[' + part_symbol_list[i] + ']' + patch for patch in part_patches]
        part_patches_list.append(part_patches)

    # 检查小节数能否对上
    bar_equality_flag = True
    for i in range(1, len(part_symbol_list)):
        if len(part_patches_list[0]) != len(part_patches_list[i]):
            bar_equality_flag = False
            print(abc_path, 'Warning: unequal bar number')
            break
    if not bar_equality_flag:
        return

    transpose_abc_text = ''
    for j in range(len(part_patches_list[0])):
        for i in range(len(part_symbol_list)):
            transpose_abc_text = transpose_abc_text + part_patches_list[i][j].strip()
        transpose_abc_text += '\n'

    transpose_abc_text = ''.join(metadata_lines) + transpose_abc_text

    if addR:
        transposed_abc_path = os.path.join(DES_DIR + '_r', abc_name + '.abc')
    else:
        transposed_abc_path = os.path.join(DES_DIR, abc_name + '.abc')
    with open(transposed_abc_path, 'w', encoding='utf-8') as w:
        w.write(transpose_abc_text)


def split_list_by_cpu(lst: list):
    num_cpus = os.cpu_count()
    split_lists = [[] for _ in range(num_cpus)]
    index = 0

    for item in lst:
        split_lists[index].append(item)
        index = (index + 1) % num_cpus

    return split_lists, num_cpus


def transpose_abcs(abc_paths: list):
    if not os.path.exists(DES_DIR):
        os.mkdir(DES_DIR)
    for abc in tqdm(abc_paths):
        transposed_an_abc_file(abc_path=abc)


def transpose_abc_dataset(dataset_path=ORI_DIR):

    abc_paths = []
    for abc_path in find_all_abc(dataset_path):
        abc_paths.append(abc_path)

    batches, num_cpu = split_list_by_cpu(abc_paths)
    pool = Pool(processes=num_cpu)
    pool.map(transpose_abcs, batches)



if __name__ == '__main__':
    transpose_abc_dataset()


