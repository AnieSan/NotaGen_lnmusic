# 用于测试多轨音乐对齐率
import music21 as m21
import os


def calculate_alignment_accuracy(abc_folder):

    total_count = 0
    aligned_count = 0

    for file in os.listdir(abc_folder):
        abc_path = os.path.join(abc_folder, file)
    
        try:
            with open(abc_path, 'r', encoding='utf-8') as f:
                abc_text = f.read()

            if not '%%score' in abc_text:
                continue
            
            total_count += 1

            # 统一删掉abc的最后一行，避免有因为patch_length不够没生成全的情况

            abc_text = '\n'.join(abc_text.split('\n')[:-1])
            abc_text_lines = abc_text.split('\n')
            print(file)

            # 滤掉除 %%score 以外的 %%行
            filtered_abc_text_lines = []
            for i, line in enumerate(abc_text_lines):
                save_state = True
                if line.startswith("%%"):
                    save_state = False
                if save_state:
                    filtered_abc_text_lines.append(line)

            # 分割为各个声部
            tunebody_index = None
            metadata_index = None
            parts_symbol_list = []
            for i, line in enumerate(filtered_abc_text_lines):
                if line.startswith('V:'):
                    parts_symbol_list.append(line.split()[0].strip())
                if line.startswith('V:1'):
                    metadata_index = i
                if line.startswith('[V:1]'):
                    tunebody_index = i
                    break
            if tunebody_index is None:
                continue
            parts_symbol_list = sorted(parts_symbol_list)

            metadata_lines = filtered_abc_text_lines[:metadata_index]
            tunebody_lines = filtered_abc_text_lines[tunebody_index:]

            parts_text_list = []
            for i in range(len(parts_symbol_list)):
                parts_text_list.append('\n'.join(metadata_lines) + '\nV:1\n')

            for line in tunebody_lines:
                # 处理前n-1个声部
                for i in range(len(parts_symbol_list) - 1):
                    start_sign = '[' + parts_symbol_list[i] + ']'
                    end_sign = '[' + parts_symbol_list[i+1] + ']'
                    start_index = line.index(start_sign) + len(start_sign)
                    end_index = line.index(end_sign)
                    parts_text_list[i] = parts_text_list[i] + line[start_index : end_index]
                # 处理最后一个声部
                start_sign = '[' + parts_symbol_list[-1] + ']'
                start_index = line.index(start_sign) + len(start_sign)
                parts_text_list[-1] = parts_text_list[-1] + line[start_index :]

            equality_flag = True
            part_measure_dur = [[] for i in range(len(parts_text_list))]
            for i, part in enumerate(parts_text_list):
                try:
                    part_score = m21.converter.parse(part)
                except Exception:
                    equality_flag = False
                # part_score.show('text')
                for measure in part_score.parts[0].getElementsByClass(m21.stream.Measure):
                    part_measure_dur[i].append(measure.duration)

            # 比较小节数
            if equality_flag:
                for i in range(len(part_measure_dur) - 1):
                    if len(part_measure_dur[i]) != len(part_measure_dur[i+1]):
                        equality_flag = False

            # 比较每小节时值
            if equality_flag:
                for i in range(len(part_measure_dur) - 1):
                    for j in range(len(part_measure_dur[i])):
                        if part_measure_dur[i][j] != part_measure_dur[i + 1][j]:
                            equality_flag = False

            if equality_flag:
                aligned_count += 1
            else:
                print('Unaligned')

        except:
            continue

    print('aligned count:', aligned_count, 'total count:', total_count)


if __name__ == '__main__':
    calculate_alignment_accuracy('output/bgpt_pretrain_pianoR_beta_patchsize8_patchlen1024')