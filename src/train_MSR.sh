# 训练3个不同尺度的MSRN for SISR
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template MSRN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 96 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save MSRN_96x2 --save_results --visdom \
# --load MSRN_96x2 --resume -1
# # --ext sep \

# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template MSRN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 136 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save MSRN_136x2 --save_results --visdom \
# --load MSRN_136x2 --resume -1
# --ext sep

# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template MSRN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 192 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save MSRN_192x2 --save_results --visdom \
# --load MSRN_192x2 --resume -1
# --ext sep

# 训练3个不同尺度的FSRCNN for SISR
# Testing...
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template FSRCNN --n_GPUs 1 --lr 0.001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 20 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save FSRCNN_96x2 --save_results --visdom \
# --ext sep 

# TODO
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template FSRCNN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 136 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save FSRCNN_136x2 --save_results --visdom \
# --ext sep 

# TODO
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template FSRCNN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 192 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save FSRCNN_192x2 --save_results --visdom \
# --ext sep 

# 训练3个不同尺度的RDN for SISR
# Testing...
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template RDN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 96 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save RDN_96x2 --save_results --visdom \
# --ext sep 



# 训练3个不同尺度的EDSR for SISR
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template EDSR_paper --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 3 --scale 2 --batch_size=16 --patch_size 96 \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 \
# --save edsr_96x2 --save_results --visdom \
# --ext sep 



