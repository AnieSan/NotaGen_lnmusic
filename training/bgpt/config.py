# Configuration for generative modelling and classification
TRAIN_FOLDERS = [
                # "data/pretrain_tunesformer_transposed_data_piano/train",
                # "data/pretrain_tunesformer_transposed_data_pianoR/train"
                "data/finetune_tunesformer_transposed_data_pd2original/train"
                 ]     # Folder containing training data
EVAL_FOLDERS = [
                # "data/pretrain_tunesformer_transposed_data_piano/test",
                # "data/pretrain_tunesformer_transposed_data_pianoR/test"
                "data/finetune_tunesformer_transposed_data_pd2original/test"
                ]                                               # Folder containing evaluation data

# Configuration for the paths
PRE_WEIGHTS_PATH = "weights_bgpt_pretrain_piano_beta_patchsize16_layer9.pth"                           # Path to pre-trained weights
WEIGHTS_PATH     = "weights_bgpt_finetune_pd2original_beta_patchsize16_layer9.pth"                        # Path to save weights
LOGS_PATH        =    "logs_bgpt_finetune_pd2original_beta_patchsize16_layer9.txt"                              # Path to save logs

# Configuration for the model
PATCH_SIZE = 16                                                 # Patch Size
PATCH_LENGTH = 512                                             # Patch Length
BYTE_NUM_LAYERS = 3                                             # Number of layers in the decoder
PATCH_NUM_LAYERS = 9                                           # Number of layers in the encoder
HIDDEN_SIZE = 768                                               # Hidden Size

# Configuration for the training
NUM_EPOCHS = 24                                                 # Number of epochs to train for (if early stopping doesn't intervene)
LEARNING_RATE = 1e-5                                            # Learning rate for the optimizer
BATCH_SIZE = 1                                                  # Batch size for training
ACCUMULATION_STEPS = 1                                          # Accumulation steps to simulate large batch size
PATCH_SAMPLING_BATCH_SIZE = 0                                   # Batch size for patch during training, 0 for full conaudio
LOAD_FROM_CHECKPOINT = False                                    # Whether to load weights from a checkpoint
LOAD_FROM_PRE_CHECKPOINT = True                                 # Whether to load pre-trained weights from a checkpoint
WANDB_LOGGING = False                                           # Whether to log to wandb

# Configuration for inference
INFERENCE_WEIGHTS_PATH = "weights_bgpt_finetune_pd2original_beta_patchsize16_layer9.pth"               # Path to weights for inference
OUTPUT_FOLDER           = "output/bgpt_finetune_pd2original_beta_patchsize16_layer9"                                        # Folder to save output files
TARGET_EXT = "abc"                                              # Target extension for inference
NUM_SAMPLES = 1000                                               # Number of samples to generate (only for generate mode)
TOP_K = 8                                                       # Top k for sampling
TOP_P = 0.8                                                      # Top p for sampling
TEMPERATURE = 1.2                                                 # Temperature for sampling
