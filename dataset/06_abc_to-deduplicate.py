import os
import bisect
import json
from utils import find_all_abc



def write_close_length_abc_pairs(threshold=0.1):
    # 比较规范化文本的长度，阈值为0.1（长度在90%-111%)

    file_size_list = []
    count = 0
    for abc_file in os.listdir('06_1_abc_text-normalized/musescoreV2'):
        count += 1
        if count % 1000 == 0:
            print(count)
        abc_path = os.path.join('06_1_abc_text-normalized/musescoreV2', abc_file)
        filesize = os.path.getsize(abc_path)
        file_size_list.append((abc_file, filesize))

    file_size_list.sort(key=lambda x: x[1])
    size_list = [ele[1] for ele in file_size_list]
    file_list = [ele[0] for ele in file_size_list]

    with open('07_abc_deduplicated/00_file_list.json', 'w', encoding='utf-8') as f:
        json.dump(file_list, f)
    with open('07_abc_deduplicated/00_size_list.json', 'w', encoding='utf-8') as f:
        json.dump(size_list, f)

    # pairs = []
    # count = 0
    # for i, ref_abc_file in enumerate(file_list):
    #     count += 1
    #     if count % 1000 == 0:
    #         print(count)
    #     ref_size = size_list[i]
    #     left_size = ref_size * (1 - threshold)
    #     right_size = ref_size / (1 - threshold)
    #     left_size_position = bisect.bisect_left(size_list, left_size)
    #     right_size_position = bisect.bisect_right(size_list, right_size)
    #
    #     for j in range(left_size_position, right_size_position):
    #         if 0 <= j < len(file_list) and j != i:
    #             est_abc_file = file_list[j]
    #             pairs.append((ref_abc_file, est_abc_file))
    #
    #
    # with open('07_abc_deduplicated/01_close_length_abc_pairs_' + str(threshold) + '.json', 'w', encoding='utf-8') as f:
    #     json.dump(pairs, f)


if __name__ == '__main__':
    write_close_length_abc_pairs()