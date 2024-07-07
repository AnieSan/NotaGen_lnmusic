import os
import json
from multiprocessing import Pool
from rapidfuzz import fuzz
from tqdm import tqdm
from abctoolkit.utils import extract_metadata_and_tunebody_rotated, find_all_abc


# check_folder = 'output/bgpt_pretrain_piano_deduplicated'
# reference_jsonl = '../data/pretrain_tunesformer_transposed_data_piano_deduplicated_train.jsonl'

# check_file_text_dict = {}
# for file in os.listdir(check_folder):
#     file_path = os.path.join(check_folder, file)
#     with open(file_path, 'r', encoding='utf-8') as f:
#         check_file_text_dict[file] = f.read()

# reference_file_text_dict = {}
# with open(reference_jsonl, 'r', encoding='utf-8') as f:
#     for line in f:
#         data = json.loads(line.strip())
#         reference_file_text_dict[data['filename']] = data['output']

# print('Load finished.')


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


def write_8_bars_of_dataset(data_folder):
    eight_bar_dict = {}
    dict_path = os.path.join('data', '8_bar_dict_' + os.path.split(data_folder)[-1] + '.json')

    count = 0
    for abc_path in find_all_abc(data_folder):
    # for abc_file in os.listdir(data_folder):
        # abc_path = os.path.join(data_folder, abc_file)
        count += 1
        if count % 10000 == 0:
            print(count)
            with open(dict_path, 'w', encoding='utf-8') as w:
                json.dump(eight_bar_dict, w)

        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()

        _, tunebody_lines = extract_metadata_and_tunebody_rotated(abc_lines)

        abc_name = os.path.splitext(os.path.split(abc_path)[-1])[0]
        eight_bar_dict[abc_name] = ''.join(tunebody_lines[:8])


    with open(dict_path, 'w', encoding='utf-8') as w:
        json.dump(eight_bar_dict, w)
    




def check_plagiarism(abc_lines, data_folder):
    pass







if __name__ == '__main__':

    # split_lists, num_cpu = split_list_by_cpu(check_file_text_dict)
    # print('num_cpu', num_cpu)
    # pool = Pool(processes=num_cpu)
    # pool.map(calculate_ld_sim, split_lists)

    
    # print('Plagiarized files', len(sim_info_list))

    # with open(check_folder.split('/')[-1] + '_sim_info.txt', 'w', encoding='utf-8') as w:
    #     for info in sim_info_list:
    #         w.write(str(info) + '\n')

    write_8_bars_of_dataset("/22A052/multitrackComposer-data/10_abc_rotated/imsleeping_stringquartet")

    output_folder = 'output/weights_bgpt_llama_gpt2_imsleeping_stringquartet_keyaugment_True_patchilizer_barbyte_stream_True_p_size_16_p_length_512_p_layers_9_h_size_768_lr_1e-05_batch_1_k_8_p_0.8_temp_1.2'

    with open('data/8_bar_dict_imsleeping_stringquartet.json', 'r', encoding='utf-8') as f:
        eight_bar_dict = json.loads(f.read())

    print('Loaded')

    for abc_file in os.listdir(output_folder):
        abc_path = os.path.join(output_folder, abc_file)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_lines = f.readlines()
        
            tunebody_index = None
            for i, line in enumerate(abc_lines):
                if line.startswith('[r:'):
                    tunebody_index = i
                    break

            metadata_lines = abc_lines[:tunebody_index]
            tunebody_lines = abc_lines[tunebody_index:]

        tunebody_lines = tunebody_lines[:8]

        highest_sim = 0
        highest_key = None

        for key, value in eight_bar_dict.items():
            sim = fuzz.ratio(value, ''.join(tunebody_lines))
            if sim > highest_sim:
                highest_sim = sim
                highest_key = key
        if highest_sim > 70:
            print(abc_path, highest_key, highest_sim)

    # abc_path = os.path.join(output_folder, '20240620-114916-16.abc')
    # with open(abc_path, 'r', encoding='utf-8') as f:
    #     abc_lines = f.readlines()
    # _, tunebody_lines = extract_metadata_and_tunebody_rotated(abc_lines)
    # tunebody_lines = tunebody_lines[:8]

    # highest_sim = 0
    # highest_key = None
    # for key, value in eight_bar_dict.items():
    #     sim = fuzz.ratio(value, ''.join(tunebody_lines))
    #     if sim > highest_sim:
    #         highest_sim = sim
    #         highest_key = key
    # print(highest_key, highest_sim)