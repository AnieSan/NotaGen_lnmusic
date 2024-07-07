import torch
import random
import bisect
import re
from config import *
from transformers import GPT2Model, GPT2LMHeadModel, LlamaModel, LlamaForCausalLM, PreTrainedModel
from samplings import top_p_sampling, top_k_sampling, temperature_sampling
from abctoolkit.utils import extract_metadata_and_tunebody_rotated


class Patchilizer:
    def __init__(self):
        self.mode = PATCH_MODE
        self.stream = PATCH_STREAM
        self.delimiters = ["|:", "::", ":|", "[|", "||", "|]", "|"]
        self.regexPattern = '(' + '|'.join(map(re.escape, self.delimiters)) + ')'
        # self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        # self.mask_token_id = 3
        self.special_token_id = 0

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

                if line_bars[0] in self.delimiters:
                    new_line_bars =[line_bars[i] + line_bars[i + 1] for i in range(0, len(line_bars, 2))]
                else:
                    new_line_bars = [line_bars[0]] + [line_bars[i] + line_bars[i + 1] for i in range(1, len(line_bars), 2)]

            # skip = False
            # for i in range(len(line_bars)):
            #     if skip:
            #         skip = False
            #         continue
            #     if line_bars[i] in self.delimiters:
            #         new_line_bars.append(line_bars[i] + line_bars[i + 1])
            #         skip = True
            #     else:
            #         new_line_bars.append(line_bars[i])

            # if len(new_line_bars) > 1:
                new_line_bars[-2] += new_line_bars[-1]  # 吸收最后一个 小节线+\n 的组合
                new_line_bars = new_line_bars[:-1]
                new_bars += new_line_bars
        except:
            pass

        return new_bars

    def bytes2patches(self, abc_text, patch_size=PATCH_SIZE):
        """
        Convert a sequence of bytes into patches
        """
        bytes = [ord(c) for c in abc_text]
        if len(bytes) % patch_size != 0:
            bytes = bytes + [self.special_token_id] * (patch_size - len(bytes) % patch_size)
        patches = [bytes[i:i + patch_size] for i in range(0, len(bytes), patch_size)]
        return patches

    def patch2bytes(self, patch):
        """
        Convert a patch into a bar.
        """
        return ''.join(chr(idx) if idx > self.eos_token_id else '' for idx in patch)

    def encode(self, abc_code, patch_length=PATCH_LENGTH, patch_size=PATCH_SIZE, add_special_patches=True, generate_mode=False):
        """
        Encode music into patches of specified length.
        """
        lines = abc_code.split('\n')
        if not generate_mode:
            lines = list(filter(None, lines))  # remove empty lines

        tunebody_index = None
        for i, line in enumerate(lines):
            if line.startswith('[V:1]') or line.startswith('[r:'):
                tunebody_index = i
                break

        metadata_lines = lines[:tunebody_index]
        tunebody_lines = lines[tunebody_index:]

        metadata_lines = [line + '\n' for line in metadata_lines]
        if self.stream:
            if not generate_mode:
                tunebody_lines = ['[r:' + str(len(tunebody_lines) - bar_no) + ']' + line + '\n' for bar_no, line in
                                  enumerate(tunebody_lines)]
            else:
                tunebody_lines = [tunebody_lines[i] + '\n' for i in range(len(tunebody_lines) - 1)] + [tunebody_lines[-1]]
        else:
            tunebody_lines = [line + '\n' for line in tunebody_lines]

        if self.mode == 'byte':
            metadata_patches = self.bytes2patches(''.join(metadata_lines))
        elif self.mode == 'barbyte' or self.mode == 'linebyte':
            metadata_patches = []
            for line in metadata_lines:
                metadata_patches += self.bytes2patches(line, patch_size)

        if self.mode == 'byte':
            tunebody_patches = self.bytes2patches(''.join(tunebody_lines))
        elif self.mode == 'barbyte':
            tunebody_patches = []
            bars = self.split_bars(tunebody_lines)
            for bar in bars:
                tunebody_patches += self.bytes2patches(bar, patch_size)
        elif self.mode == 'linebyte':
            tunebody_patches = []
            for line in tunebody_lines:
                tunebody_patches += self.bytes2patches(line, patch_size)

        if add_special_patches:
            bos_patch = [self.bos_token_id] * (patch_size - 1) + [self.eos_token_id]
            eos_patch = [self.bos_token_id] + [self.eos_token_id] * (patch_size - 1)
            # bos_patch = [self.bos_token_id] * patch_size
            # eos_patch = [self.eos_token_id] * patch_size
            metadata_patches = [bos_patch] + metadata_patches
            if not generate_mode:
                tunebody_patches = tunebody_patches + [eos_patch]

        if self.stream:
            if len(metadata_patches) + len(tunebody_patches) > patch_length:
                if self.mode in ['bar', 'barbyte', 'linebyte']:
                    available_cut_indexes = [0] + [index + 1 for index, patch in enumerate(tunebody_patches) if
                                                   self.patch2bytes(patch).endswith('\n')]
                    end_index = len(metadata_patches) + len(tunebody_patches) - patch_length
                    biggest_index = bisect.bisect_right(available_cut_indexes, end_index)
                    available_cut_indexes = available_cut_indexes[:biggest_index]  # available_cut_index <= end_index
                else:
                    available_cut_indexes = range(0, len(metadata_patches) + len(tunebody_patches) - patch_length + 1)

                if len(available_cut_indexes) == 1:
                    choices = ['head']
                elif len(available_cut_indexes) == 2:
                    choices = ['head', 'tail']
                else:
                    choices = ['head', 'tail', 'middle']
                choice = random.choice(choices)
                if choice == 'head':
                    patches = metadata_patches + tunebody_patches[0:]
                elif choice == 'tail':
                    patches = metadata_patches + tunebody_patches[available_cut_indexes[-1]:]
                else:
                    cut_index = random.choice(available_cut_indexes[1:-1])
                    patches = metadata_patches + tunebody_patches[cut_index:]
            else:
                patches = metadata_patches + tunebody_patches
        else:
            patches = metadata_patches + tunebody_patches

        return patches[:patch_length]

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

        return self.base(inputs_embeds=inputs_embeds, 
                         attention_mask=target_masks,
                         labels=target_patches)

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
            tokens = patches[:,:,-(patches.shape[-1]):].squeeze(0, 1)
            tokens = torch.cat((torch.tensor([self.bos_token_id], device=self.device), tokens), dim=-1)
            patches = patches[:,:,:-(patches.shape[-1]%PATCH_SIZE)]
            # patches = patches[len(patches),  : (patches.shape[-1] // PATCH_SIZE) * PATCH_SIZE]

        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        tokens = torch.tensor([self.bos_token_id], device=self.device)
        generated_patch = []            

        while True:
            prob = self.byte_level_decoder.generate(encoded_patches[0][-1], tokens).cpu().detach().numpy()
            prob = top_k_sampling(prob, top_k=top_k, return_probs=True)
            prob = top_p_sampling(prob, top_p=top_p, return_probs=True)
            token = temperature_sampling(prob, temperature=temperature)
            generated_patch.append(token)
            # if token == self.special_token_id or len(tokens) >= PATCH_SIZE:
            if len(tokens) >= PATCH_SIZE:
                break
            else:
                tokens = torch.cat((tokens, torch.tensor([token], device=self.device)), dim=0)
        
        return generated_patch

class bGPTForClassification(PreTrainedModel):
    """
    This class is used to classify the patches generated by the bGPT model.
    It contains the patch level decoder and a classifier.
    The global average pooling is used to get the patch-level representation.
    Then, the patch-level representation is used to classify the patches.
    It inherits PreTrainedModel from transformers.
    """
    def __init__(self, encoder_config, label_size):
        super().__init__(encoder_config)
        self.patch_level_decoder = PatchLevelDecoder(encoder_config)
        self.classifier = torch.nn.Linear(encoder_config.n_embd, label_size)
        torch.nn.init.normal_(self.classifier.weight, std=0.02)

    def forward(self,
                patches: torch.Tensor):
        """
        The forward pass of the bGPT model for classification.
        :param patches: the patches to be both encoded and decoded
        :return: the logits generated by the classifier
        """
        patches = patches.reshape(len(patches), -1, PATCH_SIZE)
        
        encoded_patches = self.patch_level_decoder(patches)["last_hidden_state"]
        encoded_patches = torch.mean(encoded_patches, dim=1)
        return self.classifier(encoded_patches)
