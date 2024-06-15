import os
import json
from multiprocessing import Pool
from rapidfuzz import fuzz
from tqdm import tqdm


check_folder = 'output/bgpt_pretrain_piano_deduplicated'
reference_jsonl = '../data/pretrain_tunesformer_transposed_data_piano_deduplicated_train.jsonl'

check_file_text_dict = {}
for file in os.listdir(check_folder):
    file_path = os.path.join(check_folder, file)
    with open(file_path, 'r', encoding='utf-8') as f:
        check_file_text_dict[file] = f.read()

reference_file_text_dict = {}
with open(reference_jsonl, 'r', encoding='utf-8') as f:
    for line in f:
        data = json.loads(line.strip())
        reference_file_text_dict[data['filename']] = data['output']

print('Load finished.')


def split_list_by_cpu(check_file_text_dict):
    num_cpus = os.cpu_count()
    split_lists = [[] for _ in range(num_cpus)]
    index = 0

    for file, item in check_file_text_dict.items():
        split_lists[index].append((item, file))
        index = (index + 1) % num_cpus

    return split_lists, num_cpus



def calculate_ld_sim(abc_text_list):


    for data in abc_text_list:
        check_text, check_name = data
        is_plagiarized = False
        for name, ref_text in reference_file_text_dict.items():
            sim = fuzz.ratio(check_text, ref_text)
            if sim > 85:
                print(check_name, name, sim)
                is_plagiarized = True
                break
        if not is_plagiarized:
            print(check_name, 'not plariarized')





if __name__ == '__main__':

    split_lists, num_cpu = split_list_by_cpu(check_file_text_dict)
    print('num_cpu', num_cpu)
    pool = Pool(processes=num_cpu)
    pool.map(calculate_ld_sim, split_lists)

    
    # print('Plagiarized files', len(sim_info_list))

    # with open(check_folder.split('/')[-1] + '_sim_info.txt', 'w', encoding='utf-8') as w:
    #     for info in sim_info_list:
    #         w.write(str(info) + '\n')


