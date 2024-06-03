'''
搞很多不同形式的转置abc来实验
'''
import os
import shutil
import json
import random
from utils import find_all_abc


def filter_single_L_scores(src_folder, dst_folder):
    # 为排除部分声部因为L不一样，可能导致的对齐问题，从05_abc_transposed中过滤出只有一个L的曲子
    src_folder_path = os.path.join('05_abc_transposed', src_folder)
    dst_folder_path = os.path.join('06_abc_experiments', dst_folder)
    if not os.path.exists(dst_folder_path):
        os.mkdir(dst_folder_path)

    for abc_path in find_all_abc(src_folder_path):
        L_count = 0
        print(abc_path)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()

        for line in abc_lines:
            if line.startswith('L:'):
                L_count += 1
            if L_count > 1:
                break

        if L_count == 1:
            shutil.copy(abc_path, dst_folder_path)
        else:
            print(abc_path, 'filtered')


def extract_pd2original_files():
    # 提取pd2数据集中原调的曲谱
    for file in os.listdir('05_abc_transposed/pd2_r'):
        if 'original' in file:
            shutil.copy('05_abc_transposed/pd2_r/' + file, '05_abc_transposed/pd2original_r')


def write_dataset_jsonline():
    '''
    {
        'dataset': '...',
        'filename': '...',
        'input': '...',
        'output': '...',
    }
    '''

    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2original.jsonl', 'w', encoding='utf-8') as w:
        count = 0
        for output_path in find_all_abc('06_abc_transposed\\pd2original'):
            print(count, output_path)
            count += 1
            filename = os.path.splitext(output_path.split('\\')[-1])[0]
            dataset_folder = output_path.split('\\')[-3]
            entry_dict = {
                'dataset': '',
                'filename': '',
                'input': '',
                'output': '',
            }

            entry_dict['dataset'] = dataset_folder
            entry_dict['filename'] = filename
            entry_dict['input'] = ''

            with open(output_path, 'r', encoding='utf-8') as f:
                entry_dict['output'] = f.read()

            w.write(json.dumps(entry_dict) + '\n')


def write_data_jsonline_inputoutput():
    # 写melodyt5架构下，根据V:1生成全曲的jsonline
    '''
        {
            'dataset': '...',
            'filename': '...',
            'input': '...',
            'output': '...',
        }
    '''
    with open('jsonl_files/finetune_tunesformer_transposed_data_piano.jsonl', 'w', encoding='utf-8') as w:
        count = 0
        for abc_path in find_all_abc('05_abc_transposed\\piano'):
            print(count, abc_path)
            count += 1
            filename = os.path.splitext(abc_path.split('\\')[-1])[0]
            dataset_folder = abc_path.split('\\')[-2]
            entry_dict = {
                'dataset': '',
                'filename': '',
                'input': '',
                'output': '',
            }

            entry_dict['dataset'] = dataset_folder
            entry_dict['filename'] = filename

            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_lines = f.readlines()

            output_text = ''.join(abc_lines)
            entry_dict['output'] = output_text

            input_lines = []
            for line in abc_lines:
                if '[V:2]' in line:
                    line = line.split('[V:2]')[0] + '\n'
                input_lines.append(line)
            input_text = ''.join(input_lines)
            entry_dict['input'] = input_text

            w.write(json.dumps(entry_dict) + '\n')


def write_data_jsonline_promptoutput():
    # 写tunesformer架构下，根据V:1生成全曲的jsonline
    '''
        {
            'dataset': '...',
            'filename': '...',
            'input': '...',
            'output': '...',
        }
    '''
    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2original.jsonl', 'w', encoding='utf-8') as w:
        count = 0
        for abc_path in find_all_abc('05_abc_transposed\\pd2original'):
            print(count, abc_path)
            count += 1
            filename = os.path.splitext(abc_path.split('\\')[-1])[0]
            dataset_folder = abc_path.split('\\')[-3]
            entry_dict = {
                'dataset': '',
                'filename': '',
                'input': '',
                'output': '',
            }

            entry_dict['dataset'] = dataset_folder
            entry_dict['filename'] = filename
            entry_dict['input'] = ''

            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_lines = f.readlines()

            prompt_lines = []
            for line in abc_lines:
                if '[V:2]' in line:
                    line = line.split('[V:2]')[0] + '\n'
                prompt_lines.append(line)
            output_text = '%%prompt\n' + ''.join(prompt_lines) + '%%output\n' + ''.join(abc_lines)

            entry_dict['output'] = output_text

            w.write(json.dumps(entry_dict) + '\n')


def split_data():

    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2original.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    random.shuffle(lines)

    train_lines = lines[ : int(0.99 * len(lines))]
    validation_lines = lines[int(0.99 * len(lines)) : ]

    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2original_train.jsonl', 'w', encoding='utf-8') as train_outfile:
        train_outfile.writelines(train_lines)
    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2original_validation.jsonl', 'w', encoding='utf-8') as validation_outfile:
        validation_outfile.writelines(validation_lines)


def split_data_according_to_other_jsonl_file():
    train_file_list = []
    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2original_train.jsonl', 'r', encoding='utf-8') as f:
        ref_train_lines = f.readlines()
    for line in ref_train_lines:
        data = json.loads(line.strip())
        train_file_list.append('_'.join(data['filename'].split('_')[:2]))

    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2originalR.jsonl', 'r', encoding='utf-8') as f:
        lines = f.readlines()

    train_lines = []
    validation_lines = []

    for line in lines:
        data = json.loads(line.strip())
        if '_'.join(data['filename'].split('_')[:2]) in train_file_list:
            train_lines.append(line)
        else:
            validation_lines.append(line)

    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2originalR_train.jsonl', 'w', encoding='utf-8') as train_outfile:
        train_outfile.writelines(train_lines)
    with open('jsonl_files/finetune_tunesformer_transposed_data_pd2originalR_validation.jsonl', 'w', encoding='utf-8') as validation_outfile:
        validation_outfile.writelines(validation_lines)


if __name__ == '__main__':
    # extract_pd2original_files()
    # filter_single_L_scores('pd2', 'pd2_singleL')
    write_dataset_jsonline()
    # write_data_jsonline_inputoutput()
    # write_data_jsonline_promptoutput()
    # split_data()
    # split_data_according_to_other_jsonl_file()

    # with open('05_abc_cleaned/metadata.json', 'r', encoding='utf-8') as f:
    #     metadata_dict = json.loads(f.read())
    # 
    # for file in os.listdir('07_abc_transposed_CLAMP/piano'):
    #     try:
    #         if metadata_dict[file[:-4]]['unique'] is True:
    #             file_path = os.path.join('07_abc_transposed_CLAMP/piano', file)
    #             des_file_path = os.path.join('07_abc_transposed_CLAMP/piano_deduplicated', file)
    #             shutil.copy(file_path, des_file_path)
    #     except:
    #         continue