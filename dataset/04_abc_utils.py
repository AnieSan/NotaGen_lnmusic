import os
import re
import shutil
import jsonlines
import json
import random
import music21 as m21
from utils import find_all_abc

def first_filtering(handled_file=None):
    '''
    第一次过滤，包括以下步骤：
     1. 替换连续换行符为单个换行符
     2. 去掉小于8小节的曲子
     3. 去掉行尾小节号
     4. 去掉表示换行的$记号
    '''
    if not handled_file:
        flag = True
    else:
        flag = False

    for dataset_folder in os.listdir('04_abc_unidecoded'):
        if dataset_folder != 'musescoreV2':
            continue

        dataset_folder_path = os.path.join('04_abc_unidecoded', dataset_folder)
        filter1_dataset_folder_path = os.path.join('05_abc_cleaned', dataset_folder)

        if not os.path.exists(filter1_dataset_folder_path):
            os.mkdir(filter1_dataset_folder_path)

        for abc_file in os.listdir(dataset_folder_path):
            if abc_file == handled_file:
                flag = True
            if not flag:
                continue

            filter1_abc_path = os.path.join(filter1_dataset_folder_path, abc_file[:-4] + '.abc')
            # if os.path.exists(filter1_abc_path):
            #     continue

            abc_path = os.path.join(dataset_folder_path, abc_file)
            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_text = f.read()

            # 0. 对于“”里面有小节线的，由于会非常影响模型后续的patchilize，考虑全部删掉
            quote_re = r'"[^"]+"'
            barlines = ["|:", "::", ":|", "[|", "||", "|]", "|"]
            quote_contents = re.findall(quote_re, abc_text)
            barline_flag = False
            for quote_content in quote_contents:
                for barline in barlines:
                    if barline in quote_content:
                        barline_flag = True
            if barline_flag:
                continue

            # 1. 替换连续换行符为单个换行符
            abc_text = re.sub(r'\n+', '\n', abc_text)

            # 2. 去掉小于8小节的曲子，顺便获得总小节数
            abc_text_lines = abc_text.split('\n')
            # 匹配有行尾小节号的行
            last_bar_no = -1
            for i, line in enumerate(abc_text_lines):
                if re.search(r'%\d+$', line):
                    bar_no = int(line.split('%')[-1])
                    if bar_no > last_bar_no:
                        last_bar_no = bar_no
                    abc_text_lines[i] = re.sub(r'%\d+$', '', line)

            if -1 < last_bar_no < 8:
                print(abc_file, last_bar_no)
                continue

            # 3. 去掉$记号 和\"
            for i, line in enumerate(abc_text_lines):
                if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
                    continue
                else:
                    # if '|$' in line:
                    #     abc_text_lines[i] = abc_text_lines[i].replace('|$', '|')
                    # if ':$' in line:
                    #     abc_text_lines[i] = abc_text_lines[i].replace(':$', ':')
                    # if ']$' in line:
                    #     abc_text_lines[i] = abc_text_lines[i].replace(']$', ']')
                    if r'\"' in line:
                        abc_text_lines[i] = abc_text_lines[i].replace(r'\"', '')

            # 4. 去掉歌不必要的information field以及%%行
            useless_info_field = ['X:', 'T:', 'C:', 'W:', 'w:', 'Z:']
            filtered_abc_lines = []
            for line in abc_text_lines:
                save_flag = True
                if line[0:2] in useless_info_field:
                    save_flag = False
                if line.startswith('%%') and not line.startswith('%%score'):
                    save_flag = False
                if save_flag:
                    filtered_abc_lines.append(line)

            # 5. 保存到 04_dataset-filter1 文件夹
            abc_text = '\n'.join(filtered_abc_lines)

            with open(filter1_abc_path, 'w', encoding='utf-8') as f:
                f.write(abc_text)
                print(abc_file, 'finished')


def rename_suffix_to_abc():
    for abc_path in find_all_abc('04_abc_cleaned\\piano'):
        if not abc_path.endswith('.abc'):
            print(abc_path)
            os.rename(abc_path, abc_path[:-4] + '.abc')


def extract_MAD_abc_files():
    # 提取MAD的450w首abc文件
    count = 0
    with jsonlines.open('03_abc/MAD.jsonl') as reader:
        for data in reader:
            count += 1
            if count % 1000 == 0:
                print(count)
            filename = data['md5']
            abc_text = data['abc_sheet']
            filepath = os.path.join('03_abc/MAD', filename + '.abc')
            with open(filepath, 'w', encoding='utf-8') as w:
                w.write(abc_text)


def extract_MAD_abc_files2():
    for i in range(1, 10):
        file_path = '03_abc/MAD_' + str(i) + '.jsonl'
        item_count = 0
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                item_count += 1
                if item_count % 1000 == 0:
                    print(item_count)
                data = json.loads(line.strip())
                abc_text = data['data']
                name = data['name']
                with open(os.path.join('03_abc/MAD', name + '.abc'), 'w', encoding='utf-8') as w:
                    w.write(abc_text)


def statistics_on_information_field():

    count = 0
    MIDI_info_types = []
    for abc_path in find_all_abc('03_abc\\musescoreV2'):
        count += 1
        if count % 10000 == 0:
            print(count)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()
        for line in abc_lines:
            if line[0].isalpha() and line[1] == ':':    # informaion field
                if line[0] not in ['X', 'T', 'C', 'L', 'M', 'K', 'Q', 'V', 'W', 'w', 'Z'] and \
                        not line.startswith('I:percmap') and \
                        not line.startswith('U:s=!stemless!'):
                    print(abc_path, line)
            elif line.startswith('%%'):
                if not line.startswith('%%score'):# and not line.startswith('%%MIDI control'):
                    midi_info_type = line.split()[1]
                    if not midi_info_type in MIDI_info_types:
                        MIDI_info_types.append(midi_info_type)
                        print(abc_path, line)
    print(MIDI_info_types)


def statistics_on_Knone_field():

    count = 0
    MIDI_info_types = []
    for abc_path in find_all_abc('03_abc\\musescoreV2'):
        count += 1
        if count % 10000 == 0:
            print(count)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()
        for i, line in enumerate(abc_lines):
            if line.strip() == 'K:none':
                try:
                    if abc_lines[i-1].split()[1].strip() != 'perc':
                        print(abc_path, 'not perc')
                except:
                    print(abc_path, 'special case')


def check_abc_through_m21():
    # 用music21把musescoreV2全部过一遍，检查有无问题
    for abc_path in find_all_abc('03_abc\musescoreV2'):
        print(abc_path)
        try:
            m21.converter.parse(abc_path)
        except Exception as e:
            print(e)






if __name__ == '__main__':

    first_filtering()
    # rename_suffix_to_abc()
    # extract_MAD_abc_files()
    # extract_MAD_abc_files2()
    # statistics_on_information_field()
    # statistics_on_Knone_field()

    # check_abc_through_m21()


    # for abc_path in find_all_abc('05_abc_transposed\\pd2'):
    #     with open(abc_path, 'r', encoding='utf-8') as f:
    #         abc_text = f.read()
    #         matches = re.findall(r'\[L:[^\]]*\]', abc_text)
    #         if matches:
    #             print(abc_path)


    # for file in os.listdir('03_abc/musescoreV2/musescoreV2'):
    #     file_path = os.path.join('03_abc/musescoreV2/musescoreV2', file)
    #     print(file)
    #     shutil.move(file_path, '03_abc/musescore2')



