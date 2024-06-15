import os
import json
import random
import shutil


def extract_files():

    folder_path = 'data/pretrain_tunesformer_transposed_data_piano_deduplicated/test'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    with open('../data/pretrain_tunesformer_transposed_data_piano_deduplicated_validation.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            # 逐行读取JSONL文件，将每行的JSON字符串转换为字典
            data = json.loads(line.strip())
            filename = data['filename']
            text = data['output']
            filepath = os.path.join(folder_path, filename + '.abc')
            with open(filepath, 'w', encoding='utf-8') as w:
                w.write(text)


def split_data():

    folder = 'data/musescore_v240612'
    train_folder = 'data/musescore_v240612/train'
    eval_folder = 'data/musescore_v240612/eval'

    entries = os.listdir(folder)
    files = [entry for entry in entries if os.path.isfile(os.path.join(folder, entry))]
    random.shuffle(files)

    if not os.path.exists(train_folder):
        os.mkdir(train_folder)
    if not os.path.exists(eval_folder):
        os.mkdir(eval_folder)

    train_files = files[ : int(0.99 * len(files))]
    eval_files = files[int(0.99 * len(files)) : ]

    for file in train_files:
        src_path = os.path.join(folder, file)
        shutil.move(src_path, train_folder)
    for file in eval_files:
        src_path = os.path.join(folder, file)
        shutil.move(src_path, eval_folder)




if __name__ == '__main__':
    split_data()
    # shutil.rmtree('data/musescore_v240612')



