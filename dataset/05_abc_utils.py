import os
import re
import random
import shutil
import json
import pandas as pd
import hashlib
from collections import Counter
from rapidfuzz import fuzz
import matplotlib.pyplot as plt
from utils import (split_abc, split_bars, find_invalid_bars_idx, check_final_barline,
                   find_all_abc, ld_sim, check_bar_patch_num_equality)



def statistics_on_text_strings():
    # 经观察，数据集中包含大量text strings，格式为 "^text" / "_text"
    # 遍历数据集，统计频率，较高的应是音乐相关术语，应保留，频率较低的可以删去
    dataset_path_list = [
        '05_abc_cleaned\\musescoreV2',
    ]
    text_string_dict = {}
    
    count = 0
    for dataset_path in dataset_path_list:
        for abc_path in find_all_abc(dataset_path):
            count += 1
            if count % 1000 == 0:
                print(count)
            if count % 100000 == 0:
                with open('05_abc_cleaned/text_strings_dict.json', 'w', encoding='utf-8') as f:
                    json.dump(text_string_dict, f)
            try:
                with open(abc_path, 'r', encoding='latin-1') as f:
                    abc_text_lines = f.readlines()
                for line in abc_text_lines:
                    quote_matches = re.findall(r'"[^"]+"', line)
                    for match in quote_matches:
                        match = match.strip("\"")
                        if match[0] in ['^', '_']:
                            if match not in text_string_dict.keys():
                                text_string_dict[match] = 1
                            else:
                                text_string_dict[match] += 1
                        # if '|' in match:
                        #     print('Warning', abc_path, match)

            except:
                pass
    # 排序
    # text_string_dict = sorted(text_string_dict.items(), key=lambda x: x[1], reverse=True)
    with open('05_abc_cleaned/text_strings_dict.json', 'w', encoding='utf-8') as f:
        json.dump(text_string_dict, f)


def statistics_on_text_words():
    with open('05_abc_cleaned/text_strings_dict.json', 'r', encoding='utf-8') as f:
        string_dict = json.loads(f.read())

    text_word_dict = {}
    for string, count in string_dict.items():
        string = string.strip("\"")[1:]
        string = re.sub(r'[^a-zA-Z]', ' ', string)
        words = string.split()
        for word in words:
            word = word.lower()
            if word not in text_word_dict.keys():
                text_word_dict[word] = count
            else:
                text_word_dict[word] += count

    # 排序
    sorted_word_list = sorted(text_word_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_word_dict = {}

    for key, value in sorted_word_list:
        sorted_word_dict[key] = value

    with open('05_abc_cleaned/text_words_dict.json', 'w', encoding='utf-8') as w:
        json.dump(sorted_word_dict, w)


def filter_strings_according_to_word_count(threshold=500, threshold2=1500):
    with open('05_abc_cleaned/01_text_strings_dict.json', 'r', encoding='utf-8') as f:
        string_dict = json.loads(f.read())
    with open('05_abc_cleaned/02_text_words_dict.json', 'r', encoding='utf-8') as f:
        word_dict = json.loads(f.read())

    string_filter_dict = {}
    # string_sub_dict = {}
    for string in string_dict.keys():
        ori_string = string
        status = True
        string = string.strip("\"")[1:]
        string = re.sub(r'[^a-zA-Z]', ' ', string)
        string = ' '.join(string.split())
        if string == '' or not ori_string.isascii():
            pass
        else:
            words = string.split()
            if len(words) == 1:
                word = words[0].lower()
                if word_dict[word] < threshold2:
                    status = False
            for word in words:
                word = word.lower()
                if word_dict[word] < threshold:
                    status = False
            if status:
                if string not in string_filter_dict.keys():
                    string_filter_dict[string] = string_dict[ori_string]
                else:
                    string_filter_dict[string] += string_dict[ori_string]


    # 排序
    sorted_string_list = sorted(string_filter_dict.items(), key=lambda x: x[1], reverse=True)
    sorted_string_dict = {}

    for key, value in sorted_string_list:
        sorted_string_dict[key] = value

    with open('05_abc_cleaned/03_text_strings_filter_dict.json', 'w', encoding='utf-8') as f:
        json.dump(sorted_string_dict, f)



def filter_strings_according_to_string_count(threshold=10):
    with open('05_abc_cleaned/01_text_strings_dict.json', 'r', encoding='utf-8') as f:
        string_dict = json.loads(f.read())
    with open('05_abc_cleaned/03_text_strings_filter_dict.json', 'r', encoding='utf-8') as f:
        string_filtered_dict = json.loads(f.read())

    string_sub_dict = {}
    for string in string_dict.keys():
        # if string != "^Rall. ----":
        # if string != "^rit.   .   .    .    .     .     .      .      .":
        # if string != "^men":
        #     continue


        ori_string = string
        string = string.strip("\"")[1:]
        lookup_string = re.sub(r'[^a-zA-Z]', ' ', string)
        lookup_string = ' '.join(lookup_string.split())
        if lookup_string == '' or not ori_string.isascii():
            string_sub_dict[ori_string] = ""
        elif not lookup_string in string_filtered_dict.keys():
            string_sub_dict[ori_string] = ""
        else:
            if string_filtered_dict[lookup_string] < threshold:
                string_sub_dict[ori_string] = ""
            else:
                sub_string = string.replace('\\n', ' ')
                sub_string_subword = sub_string.split()
                if len(sub_string_subword) < 1:
                    string_sub_dict[ori_string] = ""
                else:
                    sub_string = sub_string_subword[0]
                    for i in range(1, len(sub_string_subword)):
                        if re.match(r'^[^a-zA-Z0-9]*$', sub_string_subword[i]) is not None and \
                            sub_string_subword[i-1].endswith(sub_string_subword[i]):
                            pass
                        else:
                            sub_string = sub_string + ' ' + sub_string_subword[i]

                    sub_string = ' '.join(sub_string.split())
                    # 定义匹配非字母和非数字字符的正则表达式
                    pattern = r'([^a-zA-Z0-9])\1+'
                    # 使用正则表达式将连续的非字母非数字字符替换为单个字符
                    sub_string = re.sub(pattern, r'\1', sub_string)
                    if len(sub_string) > 40:
                        string_sub_dict[ori_string] = ""
                    else:
                        string_sub_dict[ori_string] = ori_string[0] + sub_string.strip()

    with open('05_abc_cleaned/04_text_strings_sub_dict.json', 'w', encoding='utf-8') as f:
        json.dump(string_sub_dict, f)


def substitute_text_strings_in_dataset():
    with open('05_abc_cleaned/04_text_strings_sub_dict.json', 'r', encoding='utf-8') as f:
        string_sub_dict = json.loads(f.read())
    
    count = 0
    dataset_path = '05_abc_cleaned\\musescoreV2'
    dataset = dataset_path.split('\\')[-1]
    new_dataset_path = os.path.join('06_abc_text-filtered', dataset)
    if not os.path.exists(new_dataset_path):
        os.mkdir(new_dataset_path)
    
    for abc_path in find_all_abc('05_abc_cleaned\\musescoreV2'):
        abc_name = abc_path.split('\\')[-1]
        # if abc_name != '479.abc':
        #     continue
        count += 1
        if count % 1000 == 0:
            print(count)

        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text_lines = f.readlines()

        abc_text_filtered_lines = []
        for line in abc_text_lines:
            quote_matches = re.findall(r'"[^"]+"', line)
            for match in quote_matches:
                match = match.strip("\"")
                if match[0] in ['^', '_']:
                    if match not in string_sub_dict.keys():
                        raise Exception('text strings not found')
                    sub_string = string_sub_dict[match]
                    if sub_string == "":
                        line = line.replace("\"" + match + "\"", sub_string)
                    else:
                        line = line.replace(match, sub_string)
            abc_text_filtered_lines.append(line)
        
        new_abc_path = os.path.join(new_dataset_path, abc_name)
        with open(new_abc_path, 'w', encoding='utf-8') as w:
            w.writelines(abc_text_filtered_lines)


def get_normalized_text(abc_lines):

    tunebody_index = None
    for i, line in enumerate(abc_lines):
        if line == 'V:1\n':
            tunebody_index = i
            break
    if tunebody_index is None:
        raise Exception('tunebody index not found.')

    tunebody_lines = abc_lines[tunebody_index:]
    filtered_lines = []

    for line in tunebody_lines:
        quote_matches = re.findall(r'"[^"]+"', line)
        for match in quote_matches:
            if match[1] in ['^', '_']:
                line = line.replace(match, '')
        line = ''.join(line.split())
        filtered_lines.append(line)

    text = ''.join(filtered_lines)

    return text



def md5_on_dataset():

    for abc_path in find_all_abc('05_abc_cleaned\\musescoreV2'):
        count += 1
        if count % 1000 == 0:
            print(count)
        abc_name = abc_path.split('\\')[-1][:-4]
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()

        text = get_normalized_text(abc_lines)

        md5 = hashlib.md5()
        md5.update(text.encode('utf-8'))
        hash_value = md5.hexdigest()

        md5_dict[abc_name] = hash_value
        if count % 1000 == 0:
            with open('05_abc_cleaned/md5_metadata.json', 'w', encoding='utf-8') as f:
                json.dump(md5_dict, f)

    with open('05_abc_cleaned/md5_metadata.json', 'w', encoding='utf-8') as f:
        json.dump(md5_dict, f)


def write_metadata():
    with open('05_abc_cleaned/md5_metadata.json', 'r', encoding='utf-8') as f:
        md5_dict = json.loads(f.read())

    with open(r"D:\Research\Database\SymphonyNet_mxl\score.jsonl", 'r', encoding='utf-8') as f:
        score_infos = f.readlines()
    score_infos = [json.loads(line.strip()) for line in score_infos]
    score_dict = {}

    count = 0
    for info in score_infos:
        count += 1
        if count % 1000 == 0:
            print(count)
        try:
            song_id = info['url'].split('/')[-1]
            score_dict[song_id] = info['title']
        except:
            pass

    metadata_dict = {}

    count = 0
    for key, value in md5_dict.items():
        count += 1
        if count % 1000 == 0:
            print(count)
        metadata_dict[key] = {}
        metadata_dict[key]['md5'] = value
        file_path = os.path.join('06_abc_text-filtered/musescoreV2', key + '.abc')
        file_size = os.path.getsize(file_path)
        metadata_dict[key]['file_size'] = file_size
        try:
            metadata_dict[key]['title'] = score_dict[key]
        except KeyError:
            metadata_dict[key]['title'] = None
        metadata_dict[key]['unique'] = True

    with open('05_abc_cleaned/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f)


def deduplicate():
    with open('05_abc_cleaned/metadata.json', 'r', encoding='utf-8') as f:
        metadata_dict = json.loads(f.read())
    with open('05_abc_cleaned/md5_metadata.json', 'r', encoding='utf-8') as f:
        md5_dict = json.loads(f.read())

    md5_duplicate_dict = {}
    for md5 in set(md5_dict.values()):
        md5_duplicate_dict[md5] = []

    for key, info in metadata_dict.items():
        md5_duplicate_dict[info['md5']].append(key)

    repeated_titles = {}
    for md5, file_list in md5_duplicate_dict.items():
        if len(file_list) > 1:
            repeated_titles[md5] = []
            max_file_size = 0
            max_file = None
            for file in file_list:
                if metadata_dict[file]['file_size'] > max_file_size:
                    max_file_size = metadata_dict[file]['file_size']
                    max_file = file
            for file in file_list:
                if file != max_file:
                    metadata_dict[file]['unique'] = False
                    repeated_titles[md5].append(metadata_dict[file]['title'])

    with open('05_abc_cleaned/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f)

    with open('05_abc_cleaned/repeated_title.json', 'w', encoding='utf-8') as f:
        json.dump(repeated_titles, f)


def statistics_on_text_length():
    with open('05_abc_cleaned/metadata.json', 'r', encoding='utf-8') as f:
        metadata_dict = json.loads(f.read())

    count = 0
    max_length = 0
    for id, info in metadata_dict.items():
        count += 1
        if count % 1000 == 0:
            print(count)
        file_path = os.path.join('06_abc_text-filtered/musescoreV2', id + '.abc')
        with open(file_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()
        abc_text = get_normalized_text(abc_lines)
        metadata_dict[id]['text_length_for_deduplication'] = len(abc_text)
        if len(abc_text) > max_length:
            max_length = len(abc_text)

    print(max_length)

    with open('05_abc_cleaned/metadata.json', 'w', encoding='utf-8') as f:
        json.dump(metadata_dict, f)


def write_normalized_text_for_deduplication():

    with open('05_abc_cleaned/metadata.json', 'r', encoding='utf-8') as f:
        metadata_dict = json.loads(f.read())

    count = 0
    for id, info in metadata_dict.items():
        count += 1
        if count % 1000 == 0:
            print(count)

        if info['unique'] is True:
            file_path = os.path.join('06_abc_text-filtered/musescoreV2', id + '.abc')
            with open(file_path, 'r', encoding='utf-8') as f:
                abc_lines = f.readlines()
            abc_text = get_normalized_text(abc_lines)
            des_file_path = os.path.join('05_abc_text-normalized/musescoreV2', id + '.abc')
            with open(des_file_path, 'w', encoding='utf-8') as w:
                w.write(abc_text)



if __name__ == '__main__':
    # statistics_on_text_strings()
    # statistics_on_text_words()
    # filter_strings_according_to_word_count()
    # filter_strings_according_to_string_count()

    # substitute_text_strings_in_dataset()

    # md5_on_dataset()
    # write_metadata()
    # deduplicate()
    # statistics_on_text_length()
    write_normalized_text_for_deduplication()

    # with open('05_abc_cleaned/metadata.json', 'r', encoding='utf-8') as f:
    #     metadata_dict = json.loads(f.read())
    #
    # count = 0
    # len_list = [0] * ((815800 // 10) + 1)
    # for id, info in metadata_dict.items():
    #     len_index = info['file_size'] // 10
    #     if len_index < len(len_list):
    #         len_list[len_index] += 1
    #
    #
    # with open('05_abc_cleaned/filesize_statistics_10.json', 'w', encoding='utf-8') as f:
    #     json.dump(len_list, f)
    #
    # with open('05_abc_cleaned/filesize_statistics_10.json', 'r', encoding='utf-8') as f:
    #     len_list = json.loads(f.read())
    # #
    # # print(len(len_list) - 10985)
    # len_list = len_list[:len(len_list)]
    # plt.figure(figsize=(10, 6))  # 设置图形大小
    # plt.hist(len_list, bins=len(len_list) // 100, edgecolor='black')  # 绘制直方图
    # plt.xlabel('Value')  # 设置横轴标签
    # plt.ylabel('Frequency')  # 设置纵轴标签
    # plt.title('Histogram of List Values')  # 设置图形标题
    # plt.grid(True)  # 显示网格
    # plt.show()  # 显示图形
    # plt.savefig('length_dis.png', format='png')