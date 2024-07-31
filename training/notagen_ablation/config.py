import os

# --- 实验设置（共20组） -----
REDUCE_TYPE = 'none'    # 'none' | 'time' | 'voice' | 'time-voice'
PATCH_MODE = 'bar'      # 'bar' | 'barbyte' | 'linebyte' | 'byte' | 'bpe'

# --- 可能涉及更改的参数 -----
BATCH_SIZE = 8         # Batch size for training, 在H800（80G）下，bar为8，bpe为4，barbyte/linebyte/byte为25，供参考
LEARNING_RATE = 8e-5    # Learning rate for the optimizer, 可以根据batch_size按比例调整

# --- 以下参数保持固定 -----
DATA_FOLDER_DICT = {
        'none': "data/10_abc_rotated_mini",
        'time': "data/11_abc_time_reduced_mini",
        'voice': "data/11_abc_voice_reduced_mini",
        'time-voice': "data/12_abc_time-voice_reduced_mini"
}
BPE_MODEL_DICT = {
        'none': 'bpe_model/bpe_tokenizer_10_abc_rotated.model',
        'time': 'bpe_model/bpe_tokenizer_11_abc_time_reduced.model',
        'voice': 'bpe_model/bpe_tokenizer_11_abc_voice_reduced.model',
        'time-voice': 'bpe_model/bpe_tokenizer_12_abc_time-voice_reduced.model',
}

DATA_FOLDER = "data"
DATA_TRAIN_INDEX_PATH = "04_duplicated_files_musescore+midi_mini_train.jsonl" # "04_duplicated_files_musescore+midi_mini_train.jsonl"
DATA_EVAL_INDEX_PATH = "04_duplicated_files_musescore+midi_mini_eval.jsonl" # "04_duplicated_files_musescore+midi_mini_eval.jsonl"
DATA = 'musescore+midi'

EVAL_SPLIT = 0.1

PATCH_DECODER_STRUCTURE = 'gpt2'    # 'llama'
BYTE_DECODER_STRUCTURE  = 'gpt2'    # 'llama'

# Configuration for the model

PATCH_STREAM = True
PATCH_SIZE = 64 if PATCH_MODE == 'bar' else 16                  # Patch Size
if PATCH_MODE == 'bpe':
        PATCH_LENGTH = 4096
elif PATCH_MODE == 'barbyte':
        PATCH_LENGTH = 768
else:
        PATCH_LENGTH = 512                                             # Patch Length
BYTE_NUM_LAYERS = 3                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 6                                           # Number of layers in the encoder
HIDDEN_SIZE = 768                                               # Hidden Size

# Configuration for the training
NUM_EPOCHS = 64                                                 # Number of epochs to train for (if early stopping doesn't intervene)
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
LOAD_FROM_PRE_CHECKPOINT = False                                 # Whether to load pre-trained weights from a checkpoint
WANDB_LOGGING = True                                           # Whether to log to wandb

PRETRAINED_PATH = "weights_bgpt_gpt2_gpt2_musescore+midi_reduce_time-voice_patchilizer_bar_stream_True_p_size_64_p_length_512_p_layers_6_h_size_768_lr_0.0001_batch_8.pth" # Path to the pretrained weights
NAME = PATCH_DECODER_STRUCTURE + \
        "_" + BYTE_DECODER_STRUCTURE + \
        '_' + DATA + \
        '_reduce_' + REDUCE_TYPE + \
        "_patchilizer_" + PATCH_MODE + \
        "_stream_" + str(PATCH_STREAM) + \
        "_p_size_" + str(PATCH_SIZE) + \
        "_p_length_" + str(PATCH_LENGTH) + \
        "_p_layers_" + str(PATCH_NUM_LAYERS) + \
        "_h_size_" + str(HIDDEN_SIZE) + \
        "_lr_" + str(LEARNING_RATE) + \
        "_batch_" + str(BATCH_SIZE)

WEIGHTS_PATH = "weights_bgpt_" + NAME + ".pth"                  # Path to save weights
LOGS_PATH = "logs_bgpt_" + NAME + ".txt"                     # Path to save logs
WANDB_NAME = NAME

# Configuration for inference
INFERENCE_WEIGHTS_PATH = WEIGHTS_PATH               # Path to weights for inference# Folder to save output files
TARGET_EXT = "abc"                                              # Target extension for inference
NUM_SAMPLES = 100                                               # Number of samples to generate (only for generate mode)
TOP_K = 8                                                       # Top k for sampling
TOP_P = 0.8                                                      # Top p for sampling
TEMPERATURE = 1.2                                                 # Temperature for sampling
# OUTPUT_FOLDER  = "output/bgpt_pretrain_musescore_v240612_deduplicated" + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE)
OUTPUT_FOLDER = os.path.join('output', os.path.splitext(INFERENCE_WEIGHTS_PATH)[0] + '_k_' + str(TOP_K) + '_p_' + str(TOP_P) + '_temp_' + str(TEMPERATURE))
                                                                                   
