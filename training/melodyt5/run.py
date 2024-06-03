import os
os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate melodyt5 && CUDA_VISIBLE_DEVICES=0 python train.py"')
# os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate melodyt5 && python generate.py"')