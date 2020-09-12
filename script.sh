# ==========
# TRAIN
# ==========

# Kill all distributed processes
#kill $(ps aux | grep train.py | grep -v grep | awk '{print $2}')

# 4 GPUs

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=12345 train.py --opt_path option_R3_mfqev2_4G.yml

# 2 GPUs
#CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 --master_port=12354 train.py --opt_path option_R3_mfqev2_1G.yml

# 1 GPU
#CUDA_VISIBLE_DEVICES=0 python train.py --opt_path option_R3.yml

# ==========
# TEST
# ==========

