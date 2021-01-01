# 训练一个AIN 96 
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_96x2_L1 --save_results --visdom --ext sep

# # 96 * sqrt(2) = 136
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 136 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_136x2 --save_results --visdom --ext sep

# 96 * 2 = 192
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 192 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_192x2 --save_results --visdom --ext sep

# kill 
# ps -aux| grep AIN| awk '{if($11=="python")print $2"\n"}'|xargs kill -9


# # a test
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test mySet5 \
# --epochs 2 --print_every 10 --test_every 100 --n_threads 16 \
# --save AIN_test --save_results --ext sep



# MSRNint
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template MSRNint --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save MSRNint_96x2_L1 --save_results --visdom --ext sep

# RCANint
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template RCANint --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save RCANint_96x2_L1 --save_results --visdom --ext sep

# AIN2
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template AIN2 --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN2_96x2_L1 --save_results --visdom --ext sep