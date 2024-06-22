import os
import time
import json
import wandb
import torch
import random
import numpy as np
from copy import deepcopy
from utils import *
from config import *
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Config, get_scheduler, get_constant_schedule_with_warmup
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler



# Set up distributed training
world_size = int(os.environ['WORLD_SIZE']) if 'WORLD_SIZE' in os.environ else 1
global_rank = int(os.environ['RANK']) if 'RANK' in os.environ else 0
local_rank = int(os.environ['LOCAL_RANK']) if 'LOCAL_RANK' in os.environ else 0

if world_size > 1:
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend='nccl') if world_size > 1 else None
else:
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    
# 固定随机种子点
seed = 0 + global_rank
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

batch_size = BATCH_SIZE

patchilizer = Patchilizer()

patch_config = GPT2Config(num_hidden_layers=PATCH_NUM_LAYERS, 
                    max_length=PATCH_LENGTH, 
                    max_position_embeddings=PATCH_LENGTH,
                    vocab_size=1)
char_config = GPT2Config(num_hidden_layers=CHAR_NUM_LAYERS, 
                    max_length=PATCH_SIZE, 
                    max_position_embeddings=PATCH_SIZE,
                    vocab_size=128)

model = TunesFormer(patch_config, char_config, share_weights=SHARE_WEIGHTS)
model = model.to(device)
# print parameter number
print("Parameter Number: "+str(sum(p.numel() for p in model.parameters() if p.requires_grad)))

if world_size > 1:
    model = DDP(model, device_ids=[local_rank], output_device=local_rank,  find_unused_parameters=True)

scaler = GradScaler()
is_autocast = True
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)


def split_data(data, eval_ratio=EVAL_SPLIT):
    random.shuffle(data)
    split_idx = int(len(data)*eval_ratio)
    eval_set = data[:split_idx]
    train_set = data[split_idx:]
    return train_set, eval_set


def collate_batch(batch):
    input_patches, input_masks = [], []
    
    for input_patch in batch:
        input_patches.append(input_patch)
        input_masks.append(torch.tensor([1]*input_patch.shape[0]))

    input_patches = torch.nn.utils.rnn.pad_sequence(input_patches, batch_first=True, padding_value=0)
    input_masks = torch.nn.utils.rnn.pad_sequence(input_masks, batch_first=True, padding_value=0)
    return input_patches.to(device), input_masks.to(device)


def list_files_in_directory(directories):
    file_list = []

    for directory in directories:
        for root, dirs, files in os.walk(directory):
            for file in files:
                file_path = os.path.join(root, file)
                file_list.append(file_path)
    return file_list


class MyDataset(Dataset):
    def __init__(self, filenames):
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):

        filename = self.filenames[idx]
        filename = random.choice(filename)
        if KEY_AUGMENT:
            key = random.choice(['C#', 'F#', 'B', 'E', 'A', 'D', 'G', 'C', 'F', 'Bb', 'Eb', 'Ab', 'Db', 'Gb', 'Cb'])
            dataset = os.path.split(filename)[0]
            name = os.path.splitext(os.path.split(filename)[-1])[0]
            filepath = os.path.join(DATA_FOLDER, dataset, key, name + '_' + key + '.abc')
        else:
            filepath = os.path.join(DATA_FOLDER, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
        patch = patchilizer.encode(text, add_special_patches=True)
        patch = torch.tensor(patch)

        return patch

# call model with a batch of input
def process_one_batch(batch):
    input_patches, input_masks = batch
    
    loss = model(input_patches, input_masks).loss

    # Reduce the loss on GPU 0
    if world_size > 1:
        loss = loss.unsqueeze(0)
        dist.reduce(loss, dst=0)
        loss = loss / world_size
        dist.broadcast(loss, src=0)

    return loss.mean()


# do one epoch for training
def train_epoch():
    tqdm_train_set = tqdm(train_set)
    total_train_loss = 0
    iter_idx = 1
    model.train()
    train_steps = (epoch - 1) * len(train_set)

    for batch in tqdm_train_set:
        with autocast():
            loss = process_one_batch(batch)
        if loss is None or torch.isnan(loss).item():
            continue
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        lr_scheduler.step()
        model.zero_grad(set_to_none=True)
        total_train_loss += loss.item()
        tqdm_train_set.set_postfix({str(global_rank)+'_train_loss': total_train_loss / iter_idx})
        train_steps += 1

        # Log the training loss to wandb
        if global_rank == 0 and WANDB_LOGGING:
            wandb.log({"train_loss": total_train_loss / iter_idx}, step=train_steps)

        iter_idx += 1
        
    return total_train_loss / (iter_idx-1)

# do one epoch for eval
def eval_epoch():
    tqdm_eval_set = tqdm(eval_set)
    total_eval_loss = 0
    iter_idx = 1
    model.eval()
  
    # Evaluate data for one epoch
    for batch in tqdm_eval_set: 
        with torch.no_grad():
            loss = process_one_batch(batch)
            if loss==None or torch.isnan(loss).item():
                continue
            total_eval_loss += loss.item()
        tqdm_eval_set.set_postfix({'eval_loss': total_eval_loss / iter_idx})
        iter_idx += 1

    return total_eval_loss / (iter_idx-1)

# train and eval
if __name__ == "__main__":

    # Initialize wandb
    if WANDB_LOGGING and global_rank==0:
        wandb.login(key='de0a8f1601fe2960599d99554cbf0a2778d15604')
        wandb.init(project="tunesformer",
                   name=DATA+\
                        "_patchilizer_"+PATCHILIZER+\
                        "_p_size_"+str(PATCH_SIZE)+\
                        "_p_length_"+str(PATCH_LENGTH)+\
                        "_p_layers_"+str(PATCH_NUM_LAYERS)+\
                        "_h_size_"+str(HIDDEN_SIZE)+\
                        "_lr_"+str(LEARNING_RATE)+\
                        "_batch_"+str(BATCH_SIZE))

    # load data
    with open(DATA_DUPLICATION_INDEX_PATH, "r", encoding="utf-8") as f:
        print("Loading Data...")
        train_files = []
        eval_files = []
        for line in f:
            train_files.append(json.loads(line))

    if len(eval_files) == 0:
        train_files, eval_files = split_data(train_files)

    train_batch_nums = int(len(train_files) / batch_size)
    eval_batch_nums = int(len(eval_files) / batch_size)

    random.shuffle(train_files)
    random.shuffle(eval_files)

    train_set = train_files[:train_batch_nums*batch_size]
    eval_set = eval_files[:eval_batch_nums*batch_size]

    train_set = MyDataset(train_set)
    eval_set = MyDataset(eval_set)

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=local_rank)
    eval_sampler = DistributedSampler(eval_set, num_replicas=world_size, rank=local_rank)

    train_set = DataLoader(train_set, batch_size=batch_size, collate_fn=collate_batch, sampler=train_sampler, shuffle = (train_sampler is None))
    eval_set = DataLoader(eval_set, batch_size=batch_size, collate_fn=collate_batch, sampler=eval_sampler, shuffle = (eval_sampler is None))

    lr_scheduler = get_constant_schedule_with_warmup(optimizer=optimizer, num_warmup_steps=1000)

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    if LOAD_FROM_PRETRAINED and os.path.exists(PRETRAINED_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(PRETRAINED_PATH, map_location='cpu')

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you hace a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, than load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())

        print(f"Successfully Loaded Pretrained Checkpoint at Epoch {checkpoint['epoch']} with Loss {checkpoint['min_eval_loss']}")

    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    if LOAD_FROM_CHECKPOINT and os.path.exists(WEIGHTS_PATH):
        # Load checkpoint to CPU
        checkpoint = torch.load(WEIGHTS_PATH, map_location='cpu')

        # Here, model is assumed to be on GPU
        # Load state dict to CPU model first, then move the model to GPU
        if torch.cuda.device_count() > 1:
            # If you have a DataParallel model, you need to load to model.module instead
            cpu_model = deepcopy(model.module)
            cpu_model.load_state_dict(checkpoint['model'])
            model.module.load_state_dict(cpu_model.state_dict())
        else:
            # Load to a CPU clone of the model, then load back
            cpu_model = deepcopy(model)
            cpu_model.load_state_dict(checkpoint['model'])
            model.load_state_dict(cpu_model.state_dict())
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_sched'])
        pre_epoch = checkpoint['epoch']
        best_epoch = checkpoint['best_epoch']
        min_eval_loss = checkpoint['min_eval_loss']
        print("Successfully Loaded Checkpoint from Epoch %d" % pre_epoch)
        checkpoint = None
    
    else:
        pre_epoch = 0
        best_epoch = 0
        min_eval_loss = float('inf')

    for epoch in range(1+pre_epoch, NUM_EPOCHS+1):
        train_sampler.set_epoch(epoch)
        eval_sampler.set_epoch(epoch)
        print('-' * 21 + "Epoch " + str(epoch) + '-' * 21)
        train_loss = train_epoch()
        eval_loss = eval_epoch()
        if global_rank==0:
            with open(LOGS_PATH,'a') as f:
                f.write("Epoch " + str(epoch) + "\ntrain_loss: " + str(train_loss) + "\neval_loss: " +str(eval_loss) + "\ntime: " + time.asctime(time.localtime(time.time())) + "\n\n")
            if eval_loss < min_eval_loss:
                best_epoch = epoch
                min_eval_loss = eval_loss
                checkpoint = { 
                                'model': model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'lr_sched': lr_scheduler.state_dict(),
                                'epoch': epoch,
                                'best_epoch': best_epoch,
                                'min_eval_loss': min_eval_loss
                                }
                torch.save(checkpoint, WEIGHTS_PATH)
        
        if world_size > 1:
            dist.barrier()

    if global_rank==0:
        print("Best Eval Epoch : "+str(best_epoch))
        print("Min Eval Loss : "+str(min_eval_loss))

        