import os
os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgpt && accelerate launch --multi_gpu --mixed_precision fp16 train-gen.py"')
# os.system('bash -c "source /root/miniconda3/etc/profile.d/conda.sh && conda activate bgpt && python check_plagiarism.py"')
