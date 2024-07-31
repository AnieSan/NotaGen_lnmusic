import os
import re
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers
from tokenizers.pre_tokenizers import Whitespace
from abctoolkit.transpose import Key2index
import random
import json
import sentencepiece as spm
from config import PATCH_SIZE, BPE_MODEL_DICT


def text_file_iterator():
    with open('04_duplicated_files_musescore+midi_mini_train.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line.strip())
            for filename in data:
                dataset = os.path.split(filename)[0]
                name = os.path.splitext(os.path.split(filename)[-1])[0]
                key = random.choice(list(Key2index.keys()))
                filepath = os.path.join('data/10_abc_rotated_mini', dataset, key, name + '_' + key + '.abc')
                with open(filepath, 'r', encoding='utf-8') as f:
                    abc_text = f.read()
                    abc_lines = abc_text.split('\n')
                    abc_lines = list(filter(None, abc_lines))
                    for line in abc_lines:
                        yield line


def train_huggingface_bpe_tokenizer():
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = Whitespace()
    # tokenizer.pre_tokenizer = pre_tokenizers.CharDelimiterSplit('\n')

    special_tokens = ["<PAD>", "<BOS>", "<EOS>", '\n']

    trainer = trainers.BpeTrainer(special_tokens=special_tokens, vocab_size=50000)

    tokenizer.train_from_iterator(text_file_iterator(), trainer)

    tokenizer.save("bpe_line_10_abc_rotated.json")


def write_sentencepiece_corpus():
    
    with open('12_abc_time-voice_reduced_mini_bpe_corpus.txt', 'w', encoding='utf-8') as w:
        with open('04_duplicated_files_musescore+midi_mini_train.jsonl', 'r', encoding='utf-8') as j:
            for line in j:
                data = json.loads(line.strip())
                for filename in data:
                    dataset = os.path.split(filename)[0]
                    name = os.path.splitext(os.path.split(filename)[-1])[0]
                    print(name)
                    key = random.choice(list(Key2index.keys()))
                    filepath = os.path.join('data/12_abc_time-voice_reduced_mini', dataset, key, name + '_' + key + '.abc')
                    with open(filepath, 'r', encoding='utf-8') as f:
                        abc_lines = f.readlines()

                    tunebody_index = None
                    for i, line in enumerate(abc_lines):
                        if line.startswith('[V:') or line.startswith('[r:'):
                            tunebody_index = i
                            break
                
                    metadata_lines = abc_lines[ : tunebody_index]
                    tunebody_lines = abc_lines[tunebody_index : ]   # 备份未省略前的tunebody_lines
                    tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in enumerate(tunebody_lines)] 
                    
                    for line in metadata_lines + tunebody_lines:
                        w.write(line)


def train_sentencepiece_bpe():
    spm.SentencePieceTrainer.train(input='12_abc_time-voice_reduced_mini_bpe_corpus.txt', model_prefix='bpe_tokenizer_12_abc_time-voice_reduced', vocab_size=50000, model_type='bpe', user_defined_symbols=['<pad>', '<bos>', '<eos>', '\n'])


class Patchilizer:  # 仅做patching长度统计使用，无stream模式，不根据patch_length做截断，不要用于别的用途
    def __init__(self, reduce_type):
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0
        self.bpe_tokenizer = spm.SentencePieceProcessor(model_file=BPE_MODEL_DICT[reduce_type])

    def split_bars(self, body_lines):
        """
        Split a body of music into individual bars.
        """
        new_bars = []
        try:
            for line in body_lines:
                line_bars = re.split(self.regexPattern, line)
                line_bars = list(filter(None, line_bars))
                new_line_bars = []

                if len(line_bars) == 1:
                    new_line_bars = line_bars
                else:
                    if line_bars[0] in self.delimiters:
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars, 2))]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  # 吸收最后一个 小节线+\n 的组合
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def bytes2patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        """
        Convert a sequence of bytes into patches
        """
        if not generate_last:
            bytes = [ord(c) for c in abc_text]
            if len(bytes) % patch_size != 0:
                bytes = bytes + [self.eos_token_id] + [self.special_token_id] * (patch_size - len(bytes) % patch_size - 1)
            patches = [bytes[i : i + patch_size] for i in range(0, len(bytes), patch_size)]
        else:   # 生成模式下，最后一个patch不添加eos和pad，长度可以不为patch_size的整数倍
            bytes = [ord(c) for c in abc_text]
            patches = [bytes[i : i + patch_size] for i in range(0, len(bytes), patch_size)]
        return patches
    
    def bar2patch(self, abc_bar, patch_size=PATCH_SIZE):    
        # 不区分生成和训练模式，因为最后一个patch一定是完整的小节或太长而被截断的小节
        bytes = [ord(c) for c in abc_bar] + [self.eos_token_id]
        if len(bytes) < patch_size:
            bytes = bytes + [self.special_token_id] * (patch_size - len(bytes))
        else:
            if abc_bar[-1] == '\n':
                bytes = [ord(c) for c in abc_bar[:patch_size-1]] + [ord('\n')]
            else:
                bytes = bytes[:patch_size]
        return [bytes]

    def encode_metadata(self, metadata_lines, patch_mode):

        if patch_mode == 'byte':
            metadata_patches = self.bytes2patches(''.join(metadata_lines))
        elif patch_mode == 'barbyte' or patch_mode == 'linebyte':
            metadata_patches = []
            for line in metadata_lines:
                metadata_patches += self.bytes2patches(line, PATCH_SIZE)
        elif patch_mode == 'bar':
            metadata_patches = []
            for line in metadata_lines:
                metadata_patches += self.bar2patch(line, PATCH_SIZE)
        elif patch_mode == 'bpe':
            metadata_patches = self.bpe_tokenizer.encode(''.join(metadata_lines))
        
        return metadata_patches
    
    def encode_tunebody(self, tunebody_lines, patch_mode, encode_mode='train'):

        if patch_mode == 'byte':
            if encode_mode == 'train':
                tunebody_patches = self.bytes2patches(''.join(tunebody_lines))
            elif encode_mode == 'generate':
                tunebody_patches = self.bytes2patches(''.join(tunebody_lines), generate_last=True)
        elif patch_mode == 'barbyte':
            tunebody_patches = []
            bars = self.split_bars(tunebody_lines)
            if encode_mode == 'train':
                for bar in bars:
                    tunebody_patches += self.bytes2patches(bar, PATCH_SIZE)
            elif encode_mode == 'generate':
                for bar in bars[:-1]:
                    tunebody_patches += self.bytes2patches(bar, PATCH_SIZE)
                tunebody_patches += self.bytes2patches(bars[-1], PATCH_SIZE, generate_last=True)
        elif patch_mode == 'linebyte':
            tunebody_patches = []
            if encode_mode == 'train':
                for line in tunebody_lines:
                    tunebody_patches += self.bytes2patches(line, PATCH_SIZE)
            elif encode_mode == 'generate':
                for line in tunebody_lines[:-1]:
                    tunebody_patches += self.bytes2patches(line, PATCH_SIZE)
                tunebody_patches += self.bytes2patches(tunebody_lines[-1], PATCH_SIZE, generate_last=True)
        elif patch_mode == 'bar':
            tunebody_patches = []
            bars = self.split_bars(tunebody_lines)
            for bar in bars:
                tunebody_patches += self.bar2patch(bar, PATCH_SIZE)
        elif patch_mode == 'bpe':
            tunebody_patches = self.bpe_tokenizer.encode(''.join(tunebody_lines))

        return tunebody_patches

    def encode_train(self, abc_text, patch_mode, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[V:'):
                tunebody_index = i
                break

        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]

        tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                            enumerate(tunebody_lines)]    # 加[r:n/n]

        metadata_patches = self.encode_metadata(metadata_lines, patch_mode)
        tunebody_patches = self.encode_tunebody(tunebody_lines, patch_mode, encode_mode='train')

        if add_special_patches:
            if patch_mode in ['bar', 'barbyte', 'linebyte', 'byte']:
                bos_patch = [self.bos_token_id] * (patch_size - 1) + [self.eos_token_id]
                eos_patch = [self.bos_token_id] + [self.eos_token_id] * (patch_size - 1)
            elif patch_mode == 'bpe':
                bos_patch = self.bos_token_id
                eos_patch = self.eos_token_id
            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]
        
        patches = metadata_patches + tunebody_patches

        return patches







if __name__ == '__main__':

    # train_bpe_tokenizer()

    # tokenizer = Tokenizer.from_file("bpe_line_10_abc_rotated.json")
    # with open('data/test/0a1c231494cdbef763ed617b3a0525ef_A.txt', 'r', encoding='utf-8') as f:
    #     abc_text = f.read()
    
    # encoded = tokenizer.encode(abc_text)
    # print("Encoded tokens:", encoded.tokens)
    # print("Token IDs:", encoded.ids)

    # write_sentencepiece_corpus()
    train_sentencepiece_bpe()

    # patchilizer = Patchilizer(reduce_type='none')

    # for file in os.listdir('/22A052/multitrackComposer-data/10_abc_rotated/imsleeping_tchai/C'):
    #     filepath = os.path.join('/22A052/multitrackComposer-data/10_abc_rotated/imsleeping_tchai/C', file)
    #     with open(filepath, 'r', encoding='utf-8') as f:
    #         abc_text = f.read()

    #     print(len(patchilizer.encode_train(abc_text, 'bpe')))

    # sp = spm.SentencePieceProcessor(model_file='bpe_tokenizer_10_abc_rotated.model')


    # encoded = sp.encode_as_pieces('[V:1][^a^b]2 [c_f]2 _f2 [c_f]2|[V:2]F2 z F F2 z2|[V:3]E,,E,,E,,E,, E,,E,,E,,E,,|[V:4]E2 z G z2 B2|[V:5]z8|[V:6].[E,B,E].[E,B,E].[E,B,E].[E,B,E] .[E,B,E].[E,B,E].[E,B,E].[E,B,E]|[V:7]z8|')
    # print(encoded)