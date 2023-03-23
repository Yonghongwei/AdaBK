#cifar100 e200 bs128  gs  2,4,8,16
import os,time
#############################
#r18
##############

os.system("CUDA_VISIBLE_DEVICES=0   python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.1 --master_port 29501 main.py --lr 0.05 --wd 0.001   --alg sgdmbk   --epochs 200  --model r18 ")

os.system("CUDA_VISIBLE_DEVICES=1   python -m torch.distributed.launch --nproc_per_node=1  --master_addr 127.0.0.2 --master_port 29502 main.py --lr 0.001 --wd 0.5   --alg adamwbk   --epochs 200  --model r18 ")

