import os
import jellyfish
import json
import pickle
from datasketch import MinHash, MinHashLSH
from rapidfuzz import fuzz

def find_all_abc(directory):
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('.abc') or file_path.endswith('txt'):
                yield file_path


def ld_sim(str_a, str_b):
    ld = jellyfish.levenshtein_distance(str_a, str_b)
    sim = 1-ld/(max(len(str_a), len(str_b)))
    return sim


def calculate_ldsim_longtext(abc_path, dataset_jsonl_path, handled_count=0):
    dataset = {}
    with open(dataset_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset[data['filename']] = data['output']

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_text = f.read()

    abc_text_list = abc_text.split('\n\n')[:-1]

    max_data_list = []
    max_sim_list = []
    count = 0
    for abc_text in abc_text_list:
        count += 1
        if count <= handled_count:
            continue
        max_ld_sim = 0
        max_data_name = None
        for filename, data in dataset.items():
            # print(data['filename'])
            # sim = ld_sim(abc_text, data['output'])
            sim = fuzz.ratio(abc_text, data)
            if sim > max_ld_sim:
                max_ld_sim = sim
                max_data_name = filename
        print((max_data_name, max_ld_sim))
        max_data_list.append('%%' + max_data_name + '\n' + 'X:' + str(count) + '\n' + dataset[max_data_name])
        max_sim_list.append(max_ld_sim)

    with open('plagiarism_reference_ldsim.abc', 'w', encoding='utf-8') as w:
        for i, max_data in enumerate(max_data_list):
            w.write('%%' + str(max_sim_list[i]) + '\n')
            w.write(max_data + '\n')


def standatdize_abc(abc_content):
    standardized = abc_content.lower()
    standardized = "".join(standardized.split())
    return standardized


def get_minhash(abc_content, num_perm=2048):
    abc_content = standatdize_abc(abc_content)
    m = MinHash(num_perm=num_perm)
    for word in abc_content:
        m.update(word.encode('utf8'))
    return m


def calculate_minhash_longtext(abc_path, dataset_jsonl_path, handled_count=0):
    dataset_minhash = {}
    dataset = []

    with open(dataset_jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            dataset.append(data)

    count = 0
    for data in dataset:
        count += 1
        print(count)
        m = get_minhash(data['output'])
        # print(m)
        dataset_minhash[data['filename']] = {'minhash': m, 'text': data['output']}

    with open(abc_path, 'r', encoding='utf-8') as f:
        abc_text = f.read()

    abc_text_list = abc_text.split('\n\n')[:-1]

    max_data_list = []
    max_sim_list = []
    count = 0
    for abc_text in abc_text_list:
        count += 1
        if count <= handled_count:
            continue
        print(count)
        max_sim = 0
        max_data = None
        abc_m = get_minhash(abc_text)
        for filename, data in dataset_minhash.items():
            similarity = abc_m.jaccard(data['minhash'])
            if similarity > max_sim:
                max_sim = similarity
                max_data = filename
        print(count, max_data, max_sim)
        max_data_list.append('%%' + max_data + '\n' + 'X:' + str(count) + '\n' + dataset_minhash[max_data]['text'])
        max_sim_list.append(max_sim)

    with open('plagiarism_reference_minhash.abc', 'w', encoding='utf-8') as w:
        for i, max_data in enumerate(max_data_list):
            w.write('%%' + str(max_sim_list[i]) + '\n')
            w.write(max_data + '\n')


def generate_and_save_minhashes(dataset_path, num_perm=128):
    """
    生成数据集的MinHash签名并保存到文件。
    """
    file_path = dataset_path + '_minhash.pkl'

    minhashes = {}
    for abc_path in find_all_abc(dataset_path):
        filename = abc_path.split('\\')[-1]
        print(abc_path)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text = f.read()
        minhash = get_minhash(abc_text, num_perm)
        minhashes[filename] = minhash

    with open(file_path, 'wb') as f:
        pickle.dump(minhashes, f)



if __name__ == '__main__':
    # calculate_ldsim_longtext(abc_path='tunesformer/output_tunes/Sat_25_May_2024_01_03_59_pd2original_beta_patchsize96_1000.abc',
    #                          dataset_jsonl_path='data/finetune_tunesformer_transposed_data_pd2original_train.jsonl',
    #                          handled_count=8)

    calculate_minhash_longtext(abc_path='tunesformer/output_tunes/Sat_25_May_2024_01_03_59_pd2original_beta_patchsize96_1000.abc',
                             dataset_jsonl_path='data/finetune_tunesformer_transposed_data_pd2original_train.jsonl')

    # generate_and_save_minhashes(r'D:\Research\Projects\MultitrackComposer\dataset\07_abc_transposed_CLAMP\piano')

    # with open(r'D:\Research\Projects\MultitrackComposer\dataset\07_abc_transposed_CLAMP\piano_minhash.pkl', 'rb') as f:
    #     dataset_minhash = pickle.load(f)



    abc_text = '''
%%score { ( 1 4 ) | ( 2 3 ) }
L:1/8
Q:1/4=120
M:4/4
K:F
V:1 treble nm="Piano" snm="Pno."
V:4 treble
V:2 bass
V:3 bass
[V:1]!mp!"^Allegretto" D/D/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]C/C/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]=B,/B,/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]_B,/B,/.d .A>_A z/ .GF/- F/D/F/G/!fine! |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]!f!"^A" D/D/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2].D,.D, D,/.D,D,/ z/ .D,D,/ D,/D,/.D, |[V:3]x8 |[V:4]x8 |
[V:1]C/C/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2].C,.C, C,/.C,C,/ z/ .C,C,/ C,/C,/.C, |[V:3]x8 |[V:4]x8 |
[V:1]=B,/B,/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2].=B,,.B,, B,,/.B,,B,,/ z/ .B,,B,,/ B,,/B,,/.B,, |[V:3]x8 |[V:4]x8 |
[V:1]_B,/B,/.d .A>_A z/ .GF/- F/D/F/G/ |:[V:2]._B,,.B,, B,,/.B,,C,/ z/ .C,C,/ C,/C,/.C, |:[V:3]x8 |:[V:4]x8 |:
[V:1]!f!"^B" [Dd]/[Dd]/.[dd'] .[Aa]>[_A_a] z/ .[Gg][Ff]/- [Ff]/[Dd]/[Ff]/[Gg]/ |[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, |[V:3]x8 |[V:4]x8 |
[V:1][Cc]/[Cc]/.[dd'] .[Aa]>[_A_a] z/ .[Gg][Ff]/- [Ff]/[Dd]/[Ff]/[Gg]/ |[V:2]C,,C, C,,/C,,/C,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |[V:3]x8 |[V:4]x8 |
[V:1][=B,=B]/[B,B]/.[dd'] .[Aa]>[_A_a] z/ .[Gg][Ff]/- [Ff]/[Dd]/[Ff]/[Gg]/ |[V:2]=B,,,=B,, B,,,/B,,,/B,,/B,,,/- B,,,/B,,,/B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1][_B,_B]/[B,B]/.[dd'] .[Aa]>[_A_a] z/ .[Gg][Ff]/- [Ff]/[Dd]/[Ff]/[Gg]/ :|[V:2]_B,,,_B,, B,,,/B,,,/B,,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, :|[V:3]x8 :|[V:4]x8 :|
[V:1]!mf!"^C" ff/f/ z/ .ff/- f/dd/- d2 |[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, |[V:3]x8 |[V:4]x8 |
[V:1]ff/f/ z/ .g_a/- a/(3g/4a/4g/4f/d/ f/g/ z |[V:2]C,,C, C,,/C,,/C,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |[V:3]x8 |[V:4]x8 |
[V:1]ff/f/ z/ .g_a/ z/ .=ac'/- c'<a |[V:2]=B,,,=B,, B,,,/B,,,/B,,/B,,,/- B,,,/B,,,/B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1]!<(! .d'.d' d'/a/d'/c'/- c'2 g'2!<)! |[V:2]_B,,,_B,, B,,,/B,,,/B,,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |[V:3]x8 |[V:4]x8 |
[V:1]!f!"^C'" [fa][fa]/[fa]/ z/ .[fa][fa]/- [fa]/.[dg][dg]/- [dg]2 |[V:2]D,,/D,,/.D, .A,,>_A,, z/ .G,,F,,/- F,,/D,,/F,,/G,,/ |[V:3]x2 D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, |[V:4]x8 |
[V:1][fa][fa]/[fa]/ z/ .[fa][dg]/ z/ .[fa][ed']/ z/ [da]/[cg] |[V:2]C,,/C,,/.D, .A,,>_A,, z/ .G,,F,,/- F,,/D,,/F,,/G,,/ |[V:3]x2 C,,/C,,/D,/C,,/- C,,/C,,/D,/C,,/ C,,/D,,/D, |[V:4]x8 |
[V:1]d'agf c'gfe |[V:2]=B,,,/B,,,/.D, .A,,>_A,, z/ .G,,F,,/- F,,/D,,/F,,/G,,/ |[V:3]x2 =B,,,/B,,,/D,/B,,,/- B,,,/B,,,/D,/B,,,/ B,,,/D,,/D, |[V:4]=b/g/d/e/ f/d/=B/d/ z/ d/B/d/ B/G/B/G/ |
[V:1]Bc/[Bd]/ z/ .[df][ec']/- [ec']4 |:[V:2]_B,,,/B,,,/.D, .A,,>_A,, z/ .G,,F,,/- F,,/D,,/F,,/G,,/ |:[V:3]x2 B,,,/B,,,/D,/C,,/- C,,/C,,/D,/C,,/ C,,/D,,/D, |:[V:4]x8 |:
[V:1]!mf!"^D" z4 [df]/[Bd]/[df]/[eg]/ [f_a]/[eg]/[df]/[Bd]/ |[V:2]B,,,B,, B,,,/B,,,/B,,/B,,,/- B,,,/B,,,/B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1]_a/4g/4f/4d/4[df] [eg]2- [eg]2- [eg]/a=a/ |[V:2]C,,C, C,,/C,,/C,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |[V:3]x8 |[V:4]x8 |
[V:1].c'a/_a/ g/f/d/e/ fg=ac' |[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, |[V:3]x8 |[V:4]x8 |
[V:1]_d'._a a/g/f/g/- g4 |[V:2]_D,,_D, D,,/D,,/D,/_E,,/- E,,/E,,/_E,/E,,/ E,,/E,,/E, |[V:3]x8 |[V:4]x8 |
[V:1][DF][EG][FA][df] [ce]2 [Ad]2 |[V:2]B,,,B,, B,,,/B,,,/B,,/B,,,/- B,,,/B,,,/B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1][Ge]2 [Af]2 [cg]2 [Ae]2 |[V:2]C,,C, C,,/C,,/C,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |[V:3]x8 |[V:4]x8 |
[V:1][da]4 a/_a/g/_g/ f/e/_e/d/ |[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, |[V:3]x8 |[V:4]x8 |
[V:1][_A_d]4 [B_e]4 :|[V:2]_D,,_D, D,,/D,,/D,/_E,,/- E,,/E,,/_E,/E,,/ E,,/E,,/E, :|[V:3]x8 :|[V:4]x8 :|
[V:1]!mp!"^E" B,6 F2 |[V:2]B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4]x8 |
[V:1]E4 D4 |[V:2]C,,C,, C,,/C,,C,,/- C,,/C,,C,,/ C,,/C,,/C,, |[V:3]x8 |[V:4]x8 |
[V:1]F8- |[V:2]=B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4]x8 |
[V:1]F8 |[V:2]=B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4]x8 |
[V:1]_B,6 F2 |[V:2]_B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4]x8 |
[V:1]E4 D4 |[V:2]C,,C,, C,,/C,,C,,/- C,,/C,,C,,/ C,,/C,,/C,, |[V:3]x8 |[V:4]x8 |
[V:1]D4 (6:4:6D/_D/C/=B,/_B,/A,/ (6:4:6_A,/G,/_G,/F,/E,/_E,/ |[V:2]D,,D,, D,,/D,,D,,/- D,,/D,,D,,/ D,,/D,,/D,, |[V:3]x8 |[V:4]x8 |
[V:1]D,8 |[V:2]D,,D,, D,,/D,,D,,/- D,,/D,,D,,/ D,,/D,,/D,, |[V:3]x8 |[V:4]x8 |
[V:1][K:bass]"^E'" B,6 F2 |[V:2]B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4][K:bass] D,/D,/.D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ |
[V:1]E4 D4 |[V:2]C,,C,, C,,/C,,C,,/- C,,/C,,C,,/ C,,/C,,/C,, |[V:3]x8 |[V:4][I:staff +1] C,/C,/[I:staff -1].D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ |
[V:1]F8 |[V:2]=B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4][I:staff +1] =B,,/B,,/[I:staff -1].D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ |
[V:1]=B,,/G,/[DF]/D/ .[A,G]F/[_A,C]/ D/.[G,C][F,=A,]/- [F,A,]/[D,G,]/[F,A,]/[G,C]/ |[V:2]=B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4]x8 |
[V:1]"_cresc."!<(! _B,6 F2 |[V:2]_B,,,B,,, B,,,/B,,,B,,,/- B,,,/B,,,B,,,/ B,,,/B,,,/B,,, |[V:3]x8 |[V:4]D,/D,/.D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ |
[V:1]E4 D4!<)! |[V:2]C,,C,, C,,/C,,C,,/- C,,/C,,C,,/ C,,/C,,/C,, |[V:3]x8 |[V:4][I:staff +1] C,/C,/[I:staff -1].D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ |
[V:1]!f! [D,D]/D,/.[DF] .[A,E]>!>![_A,C] z/ .[G,E]!>![F,D]/- [F,D]/[D,G,]/[F,=A,]/[G,C]/ |[V:2]D,,D,, D,,/D,,D,,/- D,,/D,,D,,/ D,,/D,,/D,, |[V:3]x8 |[V:4]x8 |
[V:1]D,/D,/.[DF] .[A,E]>!>![_A,C] z/ .[G,E]!>![F,D]/- [F,D]/[D,G,]/[F,=A,]/[G,C]/ |:[V:2]D,,D,, D,,/D,,D,,/- D,,/D,,D,,/ D,,/D,,/D,, |:[V:3]x8 |:[V:4]x8 |:
[V:1]"^F"!mp! [F,B,][F,B,] B,,/[F,B,][F,B,]/- [F,B,]/[F,B,]B,,/ B,,/B,,/[F,B,] |[V:2]B,,,B,, B,,,/B,,,/B,,/B,,,/- B,,,/B,,,/B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1][G,C][G,C] C,/[G,C][G,C]/- [G,C]/[G,C]C,/ C,/C,/[G,C] |[V:2]C,,C, C,,/C,,/C,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |[V:3]x8 |[V:4]x8 |
[V:1][A,D][A,D] D,/[A,D][_A,_D]/- [A,D]/[A,D]_D,/ D,/D,/[A,D] |[V:2]D,,D, D,,/D,,/D,/_D,,/- D,,/D,,/_D,/D,,/ D,,/D,,/D, |[V:3]x8 |[V:4]x8 |
[V:1][G,C][G,C] C,/[G,C][^F,=B,]/- [F,B,]/[F,B,]=B,,/ B,,/B,,/[F,B,] |[V:2]C,,C, C,,/C,,/C,/=B,,,/- B,,,/B,,,/=B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1][=F,_B,][F,B,] B,,/[F,B,][F,B,]/- [F,B,]/[F,B,]B,,/ B,,/B,,/[F,B,] |[V:2]_B,,,_B,, B,,,/B,,,/B,,/B,,,/- B,,,/B,,,/B,,/B,,,/ B,,,/B,,,/B,, |[V:3]x8 |[V:4]x8 |
[V:1][G,C][G,C] C,/[G,C][G,C]/- [G,C]/[G,C]C,/ C,/C,/[G,C] |1[V:2]C,,C, C,,/C,,/C,/C,,/- C,,/C,,/C,/C,,/ C,,/C,,/C, |1[V:3]x8 |1[V:4]x8 |1
[V:1][A,D][A,D] D,/[A,D][A,D]/- [A,D]/[A,D]D,/ D,/D,/[A,D] |[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, |[V:3]x8 |[V:4]x8 |
[V:1][A,D][A,D] D,/[A,D][A,D]/- [A,D]/[A,D]D,/ D,/D,/[A,D] :|2[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, :|2[V:3]x8 :|2[V:4]x8 :|2
[V:1]DD z/ DD/- D/D z/ z D ||[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, ||[V:3]x8 ||[V:4]D,/D,/D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ ||
[V:1]DD z/ DD/- D/D z/ z D ||[V:2]D,,D, D,,/D,,/D,/D,,/- D,,/D,,/D,/D,,/ D,,/D,,/D, ||[V:3]x8 ||[V:4]D,/D,/D .A,>_A, z/ .G,F,/- F,/D,/F,/G,/ ||
[V:1][K:treble]"^G" B,/B,/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4][K:treble] x8 |
[V:1]C/C/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]D/D/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]D/D/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]B,/B,/.d .A>_A z/ .GF/- F/D/F/G/ |[V:2]z8 |[V:3]x8 |[V:4]x8 |
[V:1]C/C/.d .A>_A z/ .GF/- F/D/F/G/!D.C.! |][V:2]z8 |][V:3]x8 |][V:4]x8 |]
    '''

    # dataset_jsonl_path = 'data/finetune_tunesformer_transposed_data_pd2original_train.jsonl'
    # dataset_minhash = {}
    # dataset = []
    # with open(dataset_jsonl_path, 'r', encoding='utf-8') as f:
    #     for line in f:
    #         data = json.loads(line.strip())
    #         dataset.append(data)
    # for data in dataset:
    #     m = get_minhash(data['output'])
    #     # print(m)
    #     dataset_minhash[data['filename']] = m


    # max_sim = 0
    # max_data = None
    # abc_m = get_minhash(abc_text)
    # for filename, data_m in dataset_minhash.items():
    #     similarity = abc_m.jaccard(data_m)
    #     if filename == '1352796.abc':
    #         print(filename, similarity)
    #
    #     if similarity > 0.99:
    #         print(filename, similarity)
    #     if similarity > max_sim:
    #         max_sim = similarity
    #         max_data = filename
    # print(max_data, max_sim)
#
