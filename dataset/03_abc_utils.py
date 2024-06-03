import os
from unidecode import unidecode


def unidecode_dataset():

    for dataset_folder in os.listdir('03_abc'):
        if dataset_folder != 'musescoreV2':
            continue

        dataset_folder_path = os.path.join('03_abc', dataset_folder)
        unidecoded_dataset_folder_path = os.path.join('04_abc_unidecoded', dataset_folder)

        if not os.path.exists(unidecoded_dataset_folder_path):
            os.mkdir(unidecoded_dataset_folder_path)

        for abc_file in os.listdir(dataset_folder_path):

            unidecoded_abc_path = os.path.join(unidecoded_dataset_folder_path, abc_file[:-4] + '.abc')
            if os.path.exists(unidecoded_abc_path):
                continue

            abc_path = os.path.join(dataset_folder_path, abc_file)
            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_text = f.read()

            abc_text = unidecode(abc_text)

            with open(unidecoded_abc_path, 'w', encoding='utf-8') as w:
                w.write(abc_text)
                print(abc_file, 'finished')


if __name__ == '__main__':
    unidecode_dataset()
