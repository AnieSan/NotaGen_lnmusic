## 安装
pip install abctoolkit
## 批量转换格式（还未整合）
## 批量转调
```python
from abctoolkit.transpose import batch_transpose

batch_transpose(ori_folder=r'D:\Research\Projects\MultitrackComposer\dataset\06_abc_text-filtered\sample',
                des_folder=r'D:\Research\Projects\MultitrackComposer\dataset\06_abc_text-filtered\sample_transposed')
```
## 数据预处理
### CLAMP数据
```python
from abctoolkit.utils import (find_all_abc, remove_information_field, remove_bar_no_annotations, Quote_re, Barlines,
                   strip_empty_bars)
from abctoolkit.convert import unidecode_abc_lines
from abctoolkit.rotate import rotate_abc
from abctoolkit.check import check_alignment_unrotated

ORI_FOLDER = '03_abc/mmd'
DES_FOLDER = '07_abc_rotated_CLAMP/mmd'

def abc_pipeline(abc_path):

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_lines = f.readlines()

    # 去掉纯换行符
    abc_lines = [line for line in abc_lines if line.strip() != '']

    # unidecode
    abc_lines = unidecode_abc_lines(abc_lines)

    # information field
    abc_lines = remove_information_field(abc_lines=abc_lines, info_fields=['X:', 'T:', 'C:', 'W:', 'w:', 'Z:', '%%MIDI'])

    # 去掉行尾小节号
    abc_lines = remove_bar_no_annotations(abc_lines)

    # 删掉 \"
    for i, line in enumerate(abc_lines):
        if re.search(r'^[A-Za-z]:', line) or line.startswith('%'):
            continue
        else:
            if r'\"' in line:
                abc_lines[i] = abc_lines[i].replace(r'\"', '')

    # 删掉含小节线的引号文本
    for i, line in enumerate(abc_lines):
        quote_contents = re.findall(Quote_re, line)
        for quote_content in quote_contents:
            for barline in Barlines:
                if barline in quote_content:
                    line = line.replace(quote_content, '')
                    abc_lines[i] = line

    # 去头尾空白小节
    try:
        stripped_abc_lines, bar_counts = strip_empty_bars(abc_lines)
    except Exception as e:
        print(abc_path, 'Error in stripping empty bars:', e)
        return
    if stripped_abc_lines is None:
        print(abc_path, 'Failed to strip')
        return

    # 检查小节数，小于8舍弃
    if bar_counts < 8:
        print(abc_path, 'Few bars:', bar_counts)
        return

    # 省略：text_annotation 处理

    # 检查小节数、小节线、小节时值是否对齐
    _, bar_no_equal_flag, bar_dur_equal_flag = check_alignment_unrotated(abc_lines)
    if not bar_no_equal_flag:
        print(abc_path, 'Unequal bar number')
        return
    if not bar_dur_equal_flag:
        print(abc_path, 'Unequal bar duration (unaligned)')
        return

    # 转置
    try:
        rotated_abc_lines = rotate_abc(stripped_abc_lines)
    except Exception as e:
        print(abc_path, 'Error in rotating:', e)
        return
    if rotated_abc_lines is None:
        print(abc_path, 'Failed to rotate')
        return

    des_path = os.path.join(DES_FOLDER, os.path.split(abc_path)[-1])
    with open(des_path, 'w', encoding='utf-8') as w:
        w.writelines(rotated_abc_lines)


def abc_pipeline_list(abc_path_list):
    for abc_path in tqdm(abc_path_list):
        try:
            abc_pipeline(abc_path)
        except Exception as e:
            # print(abc_path, e)
            pass


def batch_abc_pipeline():

    abc_path_list = []
    count = 0
    for abc_path in find_all_abc(ORI_FOLDER):
        count += 1
        if count % 10000 == 0:
            print(count)

        if os.path.getsize(abc_path) == 0:
            continue

        file = os.path.split(abc_path)[-1]
        filename = os.path.splitext(file)[0]

        abc_path_list.append(abc_path)

    print(len(abc_path_list))

    num_cpus = os.cpu_count()
    split_lists = [[] for _ in range(num_cpus)]
    index = 0

    for abc_path in abc_path_list:
        split_lists[index].append(abc_path)
        index = (index + 1) % num_cpus

    pool = Pool(processes=num_cpus)
    pool.map(abc_pipeline_list, split_lists)


if __name__ == '__main__':
    batch_abc_pipeline()

```
