import os
import json


if __name__ == '__main__':

    folder_path = 'data/finetune_tunesformer_transposed_data_pd2original/train'
    if not os.path.exists(folder_path):
        os.mkdir(folder_path)
        
    with open('../data/finetune_tunesformer_transposed_data_pd2original_train.jsonl', 'r', encoding='utf-8') as file:
        for line in file:
            # 逐行读取JSONL文件，将每行的JSON字符串转换为字典
            data = json.loads(line.strip())
            filename = data['filename']
            text = data['output']
            filepath = os.path.join(folder_path, filename + '.abc')
            with open(filepath, 'w', encoding='utf-8') as w:
                w.write(text)


    # with open('data/irishman_validation.json', 'r', encoding='utf-8') as file:
    #     data = file.read()
    
    # data = json.loads(data)
    # for i, item in enumerate(data):
    #     filename = str(i)
    #     text = item['abc notation']
    #     filepath = os.path.join(folder_path, filename + '.abc')
    #     with open(filepath, 'w', encoding='utf-8') as w:
    #         w.write(text)

    print(len(os.listdir(folder_path)))
