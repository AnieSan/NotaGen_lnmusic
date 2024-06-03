import os
os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate melodyt5 && python -m torch.distributed.launch --nproc_per_node=8 --use_env train.py"')
# os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate melodyt5 && python generate_tunesformer.py"')
# os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate melodyt5 && CUDA_VISIBLE_DEVICES=0 python train_tunesformer.py"')
