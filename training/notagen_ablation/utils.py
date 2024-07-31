import torch
import random
import bisect
import json
import re
import sentencepiece as spm
from config import *
from transformers import GPT2Model, GPT2LMHeadModel, LlamaModel, LlamaForCausalLM, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from tokenizers import Tokenizer


class Patchilizer:
    def __init__(self):
        self.mode = PATCH_MODE
        self.stream = PATCH_STREAM
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.special_token_id = 0
        self.bpe_tokenizer = spm.SentencePieceProcessor(model_file=BPE_MODEL_DICT[REDUCE_TYPE])

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
                        new_line_bars = [line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars), 2)]
                    else:
                        new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]
                    if 'V' not in new_line_bars[-1]:
                        new_line_bars[-2] += new_line_bars[-1]  # 吸收最后一个 小节线+\n 的组合
                        new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def split_patches(self, abc_text, patch_size=PATCH_SIZE, generate_last=False):
        if not generate_last and len(abc_text) % patch_size != 0:
            abc_text += chr(self.eos_token_id)
        patches = [abc_text[i : i + patch_size] for i in range(0, len(abc_text), patch_size)]
        return patches

    def bar2patch(self, abc_bar, patch_size=PATCH_SIZE):    
        # 不区分生成和训练模式，因为最后一个patch一定是完整的小节或太长而被截断的小节
        if len(abc_bar) > patch_size:
            if abc_bar[-1] == '\n':
                patch = abc_bar[ : patch_size - 1] + '\n'
            else:
                patch = abc_bar[ : patch_size]
        elif len(abc_bar) == patch_size:
            patch = abc_bar
        else:   # < patch_size
            patch = abc_bar + chr(self.eos_token_id)
        return patch

    def patch2bytes(self, patch):
        """
        Convert a patch into a bar.
        """
        bytes = ''
        for idx in patch:
            if idx == self.eos_token_id:
                break
            if idx < self.eos_token_id:
                pass
            bytes += chr(idx)
        return bytes

    def patchilize_metadata(self, metadata_lines):

        if self.mode == 'byte':
            metadata_patches = self.split_patches(''.join(metadata_lines))
        elif self.mode == 'barbyte' or self.mode == 'linebyte':
            metadata_patches = []
            for line in metadata_lines:
                metadata_patches += self.split_patches(line)
        elif self.mode == 'bar':
            metadata_patches = []
            for line in metadata_lines:
                metadata_patches.append(self.bar2patch(line))
        elif self.mode == 'bpe':
            metadata_patches = self.bpe_tokenizer.encode_as_pieces(''.join(metadata_lines))
        
        return metadata_patches
    
    def patchilize_tunebody(self, tunebody_lines, encode_mode='train'):

        if self.mode == 'byte':
            if encode_mode == 'train':
                tunebody_patches = self.split_patches(''.join(tunebody_lines))
            elif encode_mode == 'generate':
                tunebody_patches = self.split_patches(''.join(tunebody_lines), generate_last=True)
        elif self.mode == 'barbyte':
            tunebody_patches = []
            bars = self.split_bars(tunebody_lines)
            if encode_mode == 'train':
                for bar in bars:
                    tunebody_patches += self.split_patches(bar)
            elif encode_mode == 'generate':
                for bar in bars[:-1]:
                    tunebody_patches += self.split_patches(bar)
                tunebody_patches += self.split_patches(bars[-1], generate_last=True)
        elif self.mode == 'linebyte':
            tunebody_patches = []
            if encode_mode == 'train':
                for line in tunebody_lines:
                    tunebody_patches += self.split_patches(line)
            elif encode_mode == 'generate':
                for line in tunebody_lines[:-1]:
                    tunebody_patches += self.split_patches(line)
                tunebody_patches += self.split_patches(tunebody_lines[-1], generate_last=True)
        elif self.mode == 'bar':
            tunebody_patches = []
            bars = self.split_bars(tunebody_lines)
            for bar in bars:
                tunebody_patches.append(self.bar2patch(bar))
        elif self.mode == 'bpe':
            tunebody_patches = self.bpe_tokenizer.encode_as_pieces(''.join(tunebody_lines))

        return tunebody_patches

    def encode_train(self, abc_data_dict, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        abc_text = abc_data_dict[REDUCE_TYPE]
        if REDUCE_TYPE == 'time':
            unreduced_abc_text = abc_data_dict['none'] 
        elif REDUCE_TYPE == 'time-voice':
            unreduced_abc_text = abc_data_dict['voice']

        lines = abc_text.split('\n')
        lines = list(filter(None, lines))
        lines = [line + '\n' for line in lines]

        if REDUCE_TYPE in ['time', 'time-voice']:
            unreduced_lines = unreduced_abc_text.split('\n')
            unreduced_lines = list(filter(None, unreduced_lines))
            unreduced_lines = [line + '\n' for line in unreduced_lines]

        tunebody_index = -1
        for i, line in enumerate(lines):
            if line.startswith('[V:'):
                tunebody_index = i
                break

        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]

        if self.stream:
            tunebody_lines = ['[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index - 1) + ']' + line for line_index, line in
                                enumerate(tunebody_lines)]    # 加[r:n/n]

        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='train')

        if add_special_patches:
            if self.mode in ['bar', 'barbyte', 'linebyte', 'byte']:
                bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
                eos_patch = chr(self.bos_token_id) + chr(self.eos_token_id) * (patch_size - 1)
            elif self.mode == 'bpe':
                bos_patch = self.bpe_tokenizer.id_to_piece(self.bos_token_id)
                eos_patch = self.bpe_tokenizer.id_to_piece(self.eos_token_id)
            metadata_patches = [bos_patch] + metadata_patches
            tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                if self.mode in ['bar', 'barbyte', 'linebyte']:
                    available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                    line_index_for_cut_index = list(range(len(available_cut_indexes)))  # 每个cut_index对应tunebody的哪一行
                elif self.mode == 'byte':
                    available_cut_indexes = [0] + [index for index, patch in enumerate(tunebody_patches) if '\n' in patch]
                    line_index_for_cut_index = [0]
                    cur_line_index = 0
                    total_line_count = 0
                    for cut_index in available_cut_indexes[1:]:
                        cur_line_index = total_line_count + 1
                        n_count = tunebody_patches[cut_index].count('\n')
                        total_line_count += tunebody_patches[cut_index].count('\n')
                        line_index_for_cut_index.append(cur_line_index)
                elif self.mode == 'bpe':
                    available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if patch == '\n']
                    line_index_for_cut_index = list(range(len(available_cut_indexes)))

                end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                biggest_index = bisect.bisect_left(available_cut_indexes, end_index) # biggest index 在 end_index 右面一位
                if REDUCE_TYPE in ['time', 'time-voice']:   # 出于保险，biggest_index再往右移2位（依旧不一定能保证从biggest_index切，能够容纳乐曲所有的结束部分）
                    biggest_index = min(biggest_index + 2, len(available_cut_indexes) - 1)
                available_cut_indexes = available_cut_indexes[:biggest_index + 1]

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                else:
                    if choice == 'tail':
                        cut_index = len(available_cut_indexes) - 1
                    else:
                        cut_index = random.choice(range(1, len(available_cut_indexes) - 1))

                    line_index = line_index_for_cut_index[cut_index] 

                    if REDUCE_TYPE in ['time', 'time-voice']:
                        unreduced_line_0 = '[r:' + str(line_index) + '/' + str(len(tunebody_lines) - line_index  - 1) + ']' + unreduced_lines[tunebody_index + line_index]
                        stream_tunebody_lines = [unreduced_line_0] + tunebody_lines[line_index + 1 :]
                    else:
                        stream_tunebody_lines = tunebody_lines[line_index : ]
                    
                    stream_tunebody_patches = self.patchilize_tunebody(stream_tunebody_lines, encode_mode='train')
                    if add_special_patches:
                        stream_tunebody_patches = stream_tunebody_patches + [eos_patch]
                    patches = metadata_patches + stream_tunebody_patches
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches
        
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        if self.mode in ['bar', 'barbyte', 'linebyte', 'byte']:
            for patch in patches:
                id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
                id_patches.append(id_patch)
        elif self.mode == 'bpe':
            for patch in patches:
                id_patch = self.bpe_tokenizer.piece_to_id(patch)
                id_patches.append(id_patch)

        return id_patches

    def encode_generate(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True):

        lines = abc_code.split('\n')
        lines = list(filter(None, lines))
    
        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:') or line.startswith('[r:'):
                tunebody_index = i
                break
    
        metadata_lines = lines[ : tunebody_index]
        tunebody_lines = lines[tunebody_index : ]   # 备份未省略前的tunebody_lines
    
        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not abc_code.endswith('\n'): # 如果生成结果最后一行未完结
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines))]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]
    
        metadata_patches = self.patchilize_metadata(metadata_lines)
        tunebody_patches = self.patchilize_tunebody(tunebody_lines, encode_mode='generate')
    
        if add_special_patches:
            if self.mode in ['bar', 'barbyte', 'linebyte', 'byte']:
                bos_patch = chr(self.bos_token_id) * (patch_size - 1) + chr(self.eos_token_id)
            elif self.mode == 'bpe':
                bos_patch = self.bpe_tokenizer.id_to_piece(self.bos_token_id)
            metadata_patches = [bos_patch] + metadata_patches
    
        patches = metadata_patches + tunebody_patches
        patches = patches[ : patch_length]

        # encode to ids
        id_patches = []
        if self.mode in ['bar', 'barbyte', 'linebyte', 'byte']:
            for patch in patches:
                if len(patch) < PATCH_SIZE and patch[-1] != chr(self.eos_token_id):
                    id_patch = [ord(c) for c in patch]
                else:
                    id_patch = [ord(c) for c in patch] + [self.special_token_id] * (patch_size - len(patch))
                id_patches.append(id_patch)
        elif self.mode == 'bpe':
            for patch in patches:
                id_patch = self.bpe_tokenizer.piece_to_id(patch)
                id_patches.append(id_patch)
        
        return id_patches


    def decode(self, patches):
        """
        Decode patches into music.
        """
        return ''.join(self.patch2bytes(patch) for patch in patches)


class PatchLevelDecoder(PreTrainedModel):
    """
    A Patch-level Decoder model for generating patch features in an auto-regressive manner. 
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, structure, config):
        super().__init__(config)

        self.structure = structure
        if self.structure == 'gpt2':
            self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.n_embd)
        elif self.structure == 'llama':
            self.patch_embedding = torch.nn.Linear(PATCH_SIZE * 128, config.hidden_size)
        
        torch.nn.init.normal_(self.patch_embedding.weight, std=0.02)
        
        if self.structure == 'gpt2':
            self.base = GPT2Model(config)
        elif self.structure == 'llama':
            self.base = LlamaModel(config)


    def forward(self,
                patches: torch.Tensor,
                masks=None) -> torch.Tensor:
        """
        The forward pass of the patch-level decoder model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the encoded patches
        """
        patches = torch.nn.functional.one_hot(patches, num_classes=128).to(self.dtype)
        patches = patches.reshape(len(patches), -1, PATCH_SIZE * (128))
        patches = self.patch_embedding(patches.to(self.device))

        if masks==None:
            return self.base(inputs_embeds=patches)
        else:
            return self.base(inputs_embeds=patches,
                             attention_mask=masks)

class ByteLevelDecoder(PreTrainedModel):
    """
    A Byte-level Decoder model for generating the bytes within each patch in an auto-regressive manner
    based on the encoded patch features. It inherits PreTrainedModel from transformers.
    """
    def __init__(self, structure, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1

        self.structure = structure
        if self.structure == 'gpt2':
            self.base = GPT2LMHeadModel(config)
        elif self.structure == 'llama':
            self.base = LlamaForCausalLM(config)

    def forward(self,
                encoded_patches: torch.Tensor,
                target_patches: torch.Tensor):
        """
        The forward pass of the byte-level decoder model.
        :param encoded_patches: the encoded patches
        :param target_patches: the target patches
        :return: the output of the model
        """
        # preparing the labels for model training
        target_patches = torch.cat((torch.ones_like(target_patches[:,0:1])*self.bos_token_id, target_patches), dim=1)

        target_masks = target_patches == self.special_token_id
        labels = target_patches.clone().masked_fill_(target_masks, -100)

        # masking the labels for model training
        target_masks = torch.ones_like(labels)
        target_masks = target_masks.masked_fill_(labels == -100, 0)

        # select patches
        if PATCH_SAMPLING_BATCH_SIZE!=0 and PATCH_SAMPLING_BATCH_SIZE<target_patches.shape[0]:
            indices = list(range(len(target_patches)))
            random.shuffle(indices)
            selected_indices = sorted(indices[:PATCH_SAMPLING_BATCH_SIZE])

            target_patches = target_patches[selected_indices,:]
            target_masks = target_masks[selected_indices,:]
            encoded_patches = encoded_patches[selected_indices,:]

        # get input embeddings
        if self.structure == 'gpt2':
            inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.transformer.wte.weight)
        elif self.structure == 'llama':
            inputs_embeds = torch.nn.functional.embedding(target_patches, self.base.model.embed_tokens.weight)
        
        # concatenate the encoded patches with the input embeddings
        inputs_embeds = torch.cat((encoded_patches.unsqueeze(1), inputs_embeds[:,1:,:]), dim=1)

        output = self.base(inputs_embeds=inputs_embeds, 
                         attention_mask=target_masks,
                         labels=labels)

        return output

    def generate(self,
                 encoded_patch: torch.Tensor,
                 tokens: torch.Tensor):
        """
        The generate function for generating a patch based on the encoded patch and already generated tokens.
        :param encoded_patch: the encoded patch
        :param tokens: already generated tokens in the patch
        :return: the probability distribution of next token
        """
        encoded_patch = encoded_patch.reshape(1, 1, -1)
        tokens = tokens.reshape(1, -1)

        # Get input embeddings
        if self.structure == 'gpt2':
            tokens = torch.nn.functional.embedding(tokens, self.base.transformer.wte.weight)
        elif self.structure == 'llama':
            tokens = torch.nn.functional.embedding(tokens, self.base.model.embed_tokens.weight)

        # Concatenate the encoded patch with the input embeddings
        tokens = torch.cat((encoded_patch, tokens[:,1:,:]), dim=1)
        
        # Get output from model
        outputs = self.base(inputs_embeds=tokens)
        
        # Get probabilities of next token
        probs = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1)

        return probs

class bGPTLMHeadModel(PreTrainedModel):
    """
    bGPT is a byte-level language model with a hierarchical structure.
    It includes a patch-level decoder and a byte-level decoder.
    The patch-level decoder is used to generate patch features in an auto-regressive manner.
    The byte-level decoder is used to generate the bytes within each patch in an auto-regressive manner.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_structure, decoder_structure, 
                 encoder_config, decoder_config):
        super().__init__(encoder_config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.patch_level_decoder = PatchLevelDecoder(encoder_structure, encoder_config)
        self.byte_level_decoder = ByteLevelDecoder(decoder_structure, decoder_config)

    def forward(self,
                patches: torch.Tensor,
                masks: torch.Tensor):
        """
        The forward pass of the bGPT model.
        :param patches: the patches to be encoded
        :param masks: the masks for the patches
        :return: the decoded patches
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches, masks)["last_hidden_state"]
        
        left_shift_masks = masks * (masks.flip(1).cumsum(1).flip(1) > 1)
        masks[:, 0] = 0
        
        encoded_patches = encoded_patches[left_shift_masks == 1]
        patches = patches[masks == 1]
        
        return self.byte_level_decoder(encoded_patches, patches)
        
    def generate(self,
                 patches: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        if patches.shape[-1] % PATCH_SIZE != 0:
            tokens = patches[:,:,-(patches.shape[-1]%PATCH_SIZE):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
        else:
            tokens =  torch.tensor([self.bos_token_id], device=self.device)

        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        generated_patch = []            

        while True:
            prob = self.byte_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature)
            char = chr(token)
            generated_patch.append(token)

            if len(tokens) >= PATCH_SIZE:# or token == self.eos_token_id:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch


class bpeLMHeadModel(PreTrainedModel):
    def __init__(self, structure, config):
        super().__init__(config)
        self.special_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

        if structure == 'gpt2':
            self.base = GPT2LMHeadModel(config)
        elif structure == 'llama':
            self.base = LlamaForCausalLM(config)

    def forward(self,
                input_ids: torch.Tensor,
                masks: torch.Tensor):

        target_masks = input_ids == self.special_token_id
        labels = input_ids.clone().masked_fill_(target_masks, -100)
        outputs = self.base(input_ids, attention_mask=masks, labels=labels)
        
        return outputs
        
    def generate(self,
                 input_ids: torch.Tensor,
                 top_k=0,
                 top_p=1,
                 temperature=1.0):
        """
        The generate function for generating patches based on patches.
        :param patches: the patches to be encoded
        :param top_k: the top k for sampling
        :param top_p: the top p for sampling
        :param temperature: the temperature for sampling
        :return: the generated patches
        """
        
        generated_ids = []          

        while True:
            outputs = self.base(input_ids)
            prob = torch.nn.functional.softmax(outputs.logits.squeeze(0)[-1], dim=-1).cpu().detach().numpy()
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature)
            input_ids = torch.cat((input_ids, torch.tensor([token], device=self.device).unsqueeze(0)), dim=-1)
            generated_ids.append(int(token))

            if len(input_ids[0]) >= PATCH_LENGTH or token == self.eos_token_id:
                break

        return generated_ids