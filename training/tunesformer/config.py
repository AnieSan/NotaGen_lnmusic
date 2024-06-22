import os

KEY_AUGMENT = False
if KEY_AUGMENT:
    DATA_FOLDER = "/22A052/multitrackComposer-data/10_abc_rotated"
else:
    DATA_FOLDER = r"D:\Research\Projects\MultitrackComposer\dataset\07_abc_rotated_CLAMP"
DATA_DUPLICATION_INDEX_PATH = r"D:\Research\Projects\MultitrackComposer\dataset\08_abc_deduplicated\04_duplicated_files_piano.jsonl"
DATA = os.path.splitext(os.path.split(DATA_DUPLICATION_INDEX_PATH)[-1])[0].split('_')[-1]

EVAL_SPLIT = 0.01

PATCH_MODE = 'bar'
PATCH_STREAM = True
PATCH_LENGTH = 512      # Patch Length
PATCH_SIZE = 16       # Patch Size

PATCH_NUM_LAYERS = 9         # Number of layers in the encoder
CHAR_NUM_LAYERS = 3          # Number of layers in the decoder
HIDDEN_SIZE = 768

NUM_EPOCHS = 64                # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 2e-4           # Learning rate for the optimizer
BATCH_SIZE = 20  
PATCH_SAMPLING_BATCH_SIZE = 0   # Batch size for patch during training, 0 for full context
LOAD_FROM_CHECKPOINT = False     # Whether to load weights from a checkpoint
LOAD_FROM_PRETRAINED = False    # Whether to load weights from a pretrained model
SHARE_WEIGHTS = False
WANDB_LOGGING = False                                           # Whether to log to wandb

PRETRAINED_PATH = "history/weights_tunesformer_pretrain_piano_beta.pth" # Path to the pretrained weights
WEIGHTS_PATH = "weights_tunesformer_"+DATA+\
                "_patchilizer_"+PATCH_MODE+\
                '_stream_'+str(PATCH_STREAM)+\
                "_p_size_"+str(PATCH_SIZE)+\
                "_p_length_"+str(PATCH_LENGTH)+\
                "_p_layers_"+str(PATCH_NUM_LAYERS)+\
                "_h_size_"+str(HIDDEN_SIZE)+\
                "_lr_"+str(LEARNING_RATE)+\
                "_batch_"+str(BATCH_SIZE)+".pth"                 # Path to save weights
LOGS_PATH = "logs_tunesformer_"+DATA+\
            "_patchilizer_"+PATCH_MODE+\
            '_stream_'+str(PATCH_STREAM)+\
            "_p_size_"+str(PATCH_SIZE)+\
            "_p_length_"+str(PATCH_LENGTH)+\
            "_p_layers_"+str(PATCH_NUM_LAYERS)+\
            "_h_size_"+str(HIDDEN_SIZE)+\
            "_lr_"+str(LEARNING_RATE)+\
            "_batch_"+str(BATCH_SIZE)+".txt"                     # Path to save logs
