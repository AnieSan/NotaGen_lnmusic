# 用于测试多轨音乐对齐率
import music21 as m21
from fractions import Fraction
import re
import os


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



def calculate_duration(bar_text):
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
    try:
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

    return bar_duration



def calculate_alignment_accuracy_longtext(abc_path):
    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_text = f.read()

    abc_text_list = abc_text.split('\n\n')[:-1]
    abc_multitrack_text_list = []
    # 筛出多轨的，加入 abc_multitrack_text_list
    for abc in abc_text_list:
        if '%%score' in abc:
            abc_multitrack_text_list.append(abc)

    # 统一删掉abc的最后一行，避免有因为patch_length不够没生成全的情况
    for i, abc in enumerate(abc_multitrack_text_list):
        abc_multitrack_text_list[i] = '\n'.join(abc.split('\n')[:-1])

    total_count = len(abc_multitrack_text_list)
    aligned_count = 0

    for abc in abc_multitrack_text_list:
        abc_text_lines = abc.split('\n')
        print(abc_text_lines[0])

        # 滤掉除 %%score 以外的 %%行
        filtered_abc_text_lines = []
        for i, line in enumerate(abc_text_lines):
            save_state = True
            if line.startswith("%%"):
                save_state = False
            if save_state:
                filtered_abc_text_lines.append(line)

        # 分割为各个声部
        tunebody_index = None
        metadata_index = None
        parts_symbol_list = []
        for i, line in enumerate(filtered_abc_text_lines):
            if line.startswith('V:'):
                parts_symbol_list.append(line.split()[0].strip())
            if line.startswith('V:1'):
                metadata_index = i
            if line.startswith('[V:1]'):
                tunebody_index = i
                break
        if tunebody_index is None:
            continue
        parts_symbol_list = sorted(parts_symbol_list)

        metadata_lines = filtered_abc_text_lines[:metadata_index]
        tunebody_lines = filtered_abc_text_lines[tunebody_index:]

        parts_text_list = []
        for i in range(len(parts_symbol_list)):
            parts_text_list.append('\n'.join(metadata_lines) + '\nV:1\n')

        equality_flag = True
        for line in tunebody_lines:
            try:
                line_bar_list = []
                # 处理前n-1个声部
                for i in range(len(parts_symbol_list) - 1):
                    start_sign = '[' + parts_symbol_list[i] + ']'
                    end_sign = '[' + parts_symbol_list[i+1] + ']'
                    start_index = line.index(start_sign) + len(start_sign)
                    end_index = line.index(end_sign)
                    line_bar_list.append(line[start_index : end_index])
                # 处理最后一个声部
                start_sign = '[' + parts_symbol_list[-1] + ']'
                start_index = line.index(start_sign) + len(start_sign)
                line_bar_list.append(line[start_index :])

                ref_duration = calculate_duration(line_bar_list[0])
                for i in range(1, len(line_bar_list)):
                    if ref_duration != calculate_duration(line_bar_list[i]):
                        equality_flag = False
                        break
            except:
                equality_flag = False

        if equality_flag:
            aligned_count += 1
        else:
            print('Unaligned')

    print('aligned count:', aligned_count, 'total count:', total_count)


def calculate_alignment_accuracy_folder(abc_folder):
    total_count = 0
    aligned_count = 0

    for file in os.listdir(abc_folder):
        abc_path = os.path.join(abc_folder, file)

        try:
            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_text = f.read()

            if not '%%score' in abc_text:
                continue

            total_count += 1

            # 统一删掉abc的最后一行，避免有因为patch_length不够没生成全的情况

            abc_text = '\n'.join(abc_text.split('\n')[:-1])
            abc_text_lines = abc_text.split('\n')
            print(file)

            # 滤掉除 %%score 以外的 %%行
            filtered_abc_text_lines = []
            for i, line in enumerate(abc_text_lines):
                save_state = True
                if line.startswith("%%"):
                    save_state = False
                if save_state:
                    filtered_abc_text_lines.append(line)

            # 分割为各个声部
            tunebody_index = None
            metadata_index = None
            parts_symbol_list = []
            for i, line in enumerate(filtered_abc_text_lines):
                if line.startswith('V:'):
                    parts_symbol_list.append(line.split()[0].strip())
                if line.startswith('V:1'):
                    metadata_index = i
                if line.startswith('[V:1]'):
                    tunebody_index = i
                    break
            if tunebody_index is None:
                continue
            parts_symbol_list = sorted(parts_symbol_list)

            metadata_lines = filtered_abc_text_lines[:metadata_index]
            tunebody_lines = filtered_abc_text_lines[tunebody_index:]

            parts_text_list = []
            for i in range(len(parts_symbol_list)):
                parts_text_list.append('\n'.join(metadata_lines) + '\nV:1\n')

            equality_flag = True
            for line in tunebody_lines:
                line_bar_list = []
                # 处理前n-1个声部
                for i in range(len(parts_symbol_list) - 1):
                    start_sign = '[' + parts_symbol_list[i] + ']'
                    end_sign = '[' + parts_symbol_list[i + 1] + ']'
                    start_index = line.index(start_sign) + len(start_sign)
                    end_index = line.index(end_sign)
                    line_bar_list.append(line[start_index: end_index])
                # 处理最后一个声部
                start_sign = '[' + parts_symbol_list[-1] + ']'
                start_index = line.index(start_sign) + len(start_sign)
                line_bar_list.append(line[start_index:])

                ref_duration = calculate_duration(line_bar_list[0])
                for i in range(1, len(line_bar_list)):
                    if ref_duration != calculate_duration(line_bar_list[i]):
                        equality_flag = False
                        break

            if equality_flag:
                aligned_count += 1
            else:
                print('Unaligned')
        except:
            continue

    print('aligned count:', aligned_count, 'total count:', total_count)





if __name__ == '__main__':
    calculate_alignment_accuracy_longtext(r'D:\Research\Projects\MultitrackComposer\training\tunesformer\output_tunes\Wed_29_May_2024_14_36_40_weights_tunesformer_pretrain_piano_beta_patchsize96.abc')
    # calculate_alignment_accuracy_folder(r'D:\Research\Projects\MultitrackComposer\training\bgpt\output\bgpt_finetune_pd2original_beta_patchsize16_layer9')