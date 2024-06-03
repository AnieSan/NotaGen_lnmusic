PATCH_LENGTH = 256      # Patch Length
PATCH_SIZE = 96       # Patch Size

PATCH_NUM_LAYERS = 9         # Number of layers in the encoder
CHAR_NUM_LAYERS = 3          # Number of layers in the decoder

NUM_EPOCHS = 24                # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 1e-5           # Learning rate for the optimizer
BATCH_SIZE = 1  
PATCH_SAMPLING_BATCH_SIZE = 0   # Batch size for patch during training, 0 for full context
LOAD_FROM_CHECKPOINT = False     # Whether to load weights from a checkpoint
LOAD_FROM_PRETRAINED = False    # Whether to load weights from a pretrained model
SHARE_WEIGHTS = False            # Whether to share weights between the encoder and decoder

TRAIN_DATA_PATH      = '../data/finetune_tunesformer_transposed_data_pd2original_train.jsonl'
VALIDATION_DATA_PATH = '../data/finetune_tunesformer_transposed_data_pd2original_validation.jsonl'

PRETRAINED_PATH = "weights_tunesformer_pretrain_piano_beta_patchsize96.pth" # Path to the pretrained weights
WEIGHTS_PATH = "weights_tunesformer_finetune_pd2original_beta_patchsize96.pth"    # Path to save the weights
LOGS_PATH       = "logs_tunesformer_finetune_pd2original_beta_patchsize96.txt"          # Path to save the logs