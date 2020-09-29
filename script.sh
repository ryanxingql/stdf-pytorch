# ==========
# TRAIN
# ==========

# Kill all distributed processes
#kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')

# 4 GPUs
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py --opt_path option_R3_mfqev2_4G.yml

# 2 GPUs
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_R3_mfqev2_2G.yml

# 1 GPU
#CUDA_VISIBLE_DEVICES=0 python train.py --opt_path option_R3_mfqev2_1G.yml

# ==========
# TEST
# ==========

# use 1 GPU, no matter _1G.yml, _2G.yml or _4G.yml. 
# YAMLs for training and test are better the same to ensure the same exp_name and network structure.
CUDA_VISIBLE_DEVICES=0 python test.py --opt_path option_R3_mfqev2_4G.yml
