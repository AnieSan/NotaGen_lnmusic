import os
from abc_transposition import transpose_an_abc_text
from utils import find_all_abc
from multiprocessing import Pool
from tqdm import tqdm

KEY_CHOICES = ["Cb", "Gb", "Db", "Ab", "Eb", "Bb", "F", "C", "G", "D", "A", "E", "B", "F#", "C#"]

ORI_DIR = r'D:\Research\Projects\MultitrackComposer\dataset\06_abc_text-filtered\musescoreV2'
AUGMENTED_DIR = r'D:\Research\Projects\MultitrackComposer\dataset\08_abc_key-augmented\musescoreV2'

def split_list_by_cpu(lst: list):
    num_cpus = os.cpu_count()
    split_lists = [[] for _ in range(num_cpus)]
    index = 0

    for item in lst:
        split_lists[index].append(item)
        index = (index + 1) % num_cpus

    return split_lists, num_cpus


def key_augment_an_abc_file(abc_path: str):
    for key in KEY_CHOICES:
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text_lines = f.readlines()

        ori_filename = abc_path.split('\\')[-1][:-4]

        try:
            transposed_abc_text, ori_key, des_key = transpose_an_abc_text(abc_text_lines, key)
            if ori_key == des_key:
                filename = ori_filename + '_original_' + des_key
            else:
                filename = ori_filename + '_transposed_' + des_key
            augmented_filepath = os.path.join(AUGMENTED_DIR, filename + '.abc')
            with open(augmented_filepath, 'w', encoding='utf-8') as f:
                f.write(transposed_abc_text)
        except Exception as e:
            print(abc_path, e)


def key_augment_abcs(abc_paths: list):
    if not os.path.exists(AUGMENTED_DIR):
        os.mkdir(AUGMENTED_DIR)
    for abc in tqdm(abc_paths):
        key_augment_an_abc_file(abc_path=abc)


def key_augment_abc_dataset(dataset_path=ORI_DIR, augmented_dataset_path=AUGMENTED_DIR):

    abc_paths = []
    for abc_path in find_all_abc(dataset_path):
        abc_paths.append(abc_path)

    batches, num_cpu = split_list_by_cpu(abc_paths)
    pool = Pool(processes=num_cpu)
    pool.map(key_augment_abcs, batches)


if __name__ == '__main__':
    key_augment_abc_dataset()