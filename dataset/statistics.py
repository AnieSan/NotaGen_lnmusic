import os
import re
import json
import pandas as pd
from unidecode import unidecode


def find_all_abc(directory):
    for root, directories, files in os.walk(directory):
        for filename in files:
            file_path = os.path.join(root, filename)
            if file_path.endswith('.abc') or file_path.endswith('txt'):
                yield file_path


class Patchilizer:
    """
    A class for converting music bars to patches and vice versa.
    """
    def __init__(self):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def split_bars(self, body):
        """
        Split a body of music into individual bars.
        """
        bars = re.split(self.regexPattern, ''.join(body))
        bars = list(filter(None, bars))  # remove empty strings
        if bars[0] in self.delimiters:
            bars[1] = bars[0] + bars[1]
            bars = bars[1:]
        bars = [bars[i * 2] + bars[i * 2 + 1] for i in range(len(bars) // 2)]
        return bars

    def encode(self, abc_code):
        """
        Encode music into patches of specified length.
        """
        lines = unidecode(abc_code).split('\n')
        lines = list(filter(None, lines))  # remove empty lines

        body = ""
        patches = []

        for line in lines:
            if len(line) > 1 and ((line[0].isalpha() and line[1] == ':') or line.startswith('%%')):
                if body:
                    bars = self.split_bars(body)
                    patches.extend(bar + '\n' if idx == len(bars) - 1 else bar for idx, bar in enumerate(bars))
                    body = ""
                patches.append(line + '\n')
            else:
                body += line + '\n'

        if body:
            patches.extend(bar for bar in self.split_bars(body))

        return patches


patchilizer = Patchilizer()

# musescore-dataset total patch count: 646,443,664
# piano-musescore-dataset total patch count: 55,286,235
def statistics_on_total_token_counts():

    patch_tokens = 0
    file_count = 0
    for abc_path in find_all_abc('04_abc_cleaned\\piano'):
        file_count += 1
        if file_count % 5000 == 0:
            print(file_count, abc_path)
        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text = f.read()
        abc_patches = patchilizer.encode(abc_text)
        patch_tokens += len(abc_patches)

    print('total patch tokens:', patch_tokens)


def statistics_on_patch_size_and_length():

    patch_length_counts = pd.DataFrame(columns=['patch length', 'count'])
    patch_size_counts = pd.DataFrame(columns=['patch size', 'count'])

    for i in range(4096 + 1):
        patch_length_counts.loc[len(patch_length_counts.index)] = [i] + [0]
    for i in range(256 + 1):
        patch_size_counts.loc[len(patch_size_counts.index)] = [i] + [0]

    item_count = 0
    for abc_path in find_all_abc('05_abc_cleaned\musescoreV2'):
        item_count += 1
        if item_count % 1000 == 0:
            print(item_count)
            patch_length_counts.to_excel('patch_length_counts_musescoreV2.xlsx')
            patch_size_counts.to_excel('patch_size_counts_musescoreV2.xlsx')

        with open(abc_path, 'r', encoding='utf-8') as f:
            abc_text = f.read()
            patches = patchilizer.encode(abc_text)

            if len(patches) < 4096:
                patch_length_counts.iloc[len(patches)]['count'] += 1
            else:
                patch_length_counts.iloc[4096]['count'] += 1

            for patch in patches:
                if len(patch) < 256:
                    patch_size_counts.iloc[len(patch)]['count'] += 1
                else:
                    patch_size_counts.iloc[256]['count'] += 1



def statistics_on_MAD_patch_size_and_length():

    patch_length_counts = pd.DataFrame(columns=['patch length', 'count'])
    patch_size_counts = pd.DataFrame(columns=['patch size', 'count'])

    for i in range(4096 + 1):
        patch_length_counts.loc[len(patch_length_counts.index)] = [i] + [0]
    for i in range(256 + 1):
        patch_size_counts.loc[len(patch_size_counts.index)] = [i] + [0]

    item_count = 0
    string_len = 0
    for i in range(1, 10):
        mad_path = os.path.join('00_raw', 'MAD_' + str(i) + '.jsonl')
        with open(mad_path, 'r', encoding='utf-8') as file:
            for line in file:

                item_count += 1
                if item_count % 1000 == 0:
                    print(item_count)
                    patch_length_counts.to_excel('patch_length_counts_MAD.xlsx')
                    patch_size_counts.to_excel('patch_size_counts_MAD.xlsx')

                data = json.loads(line.strip())
                abc_text = data['data']
                string_len += len(abc_text)
                patches = patchilizer.encode(abc_text)

                if len(patches) < 4096:
                    patch_length_counts.iloc[len(patches)]['count'] += 1
                else:
                    patch_length_counts.iloc[4096]['count'] += 1

                for patch in patches:
                    if len(patch) < 256:
                        patch_size_counts.iloc[len(patch)]['count'] += 1
                    else:
                        patch_size_counts.iloc[256]['count'] += 1

    print(item_count)
    patch_length_counts.to_excel('patch_length_counts_MAD.xlsx')
    patch_size_counts.to_excel('patch_size_counts_MAD.xlsx')



if __name__ == '__main__':
    # statistics_on_total_token_counts()
    # statistics_on_patch_size_and_length()
    statistics_on_MAD_patch_size_and_length()

    # for abc_path in find_all_abc('04_abc_cleaned\\piano'):
    #     with open(abc_path, 'r', encoding='utf-8') as f:
    #         abc_text = f.read()
    #     matches1 = re.findall(r'\|\d+', abc_text)
    #     # matches2 = re.findall(r':\|\d+', abc_text)
    #     # matches = matches1 + matches2
    #     if matches1:
    #         print(abc_path, matches1)