# 训练一个AIN 96 
CUDA_VISIBLE_DEVICES=4,2 python main.py --trainer trainer --template AIN --n_GPUs 2 --lr 0.0001 --loss 1*L1 \
--data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
--data_test mySet5+myUrban12 \
--epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
--save AIN_96x2_L1 --save_results --visdom --ext sep

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




# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=8 --patch_size 48 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 6 --print_every 10 --test_every 100 --n_threads 8 \
# --save AIN_test_96x2_L1 --load AIN_test_96x2_L1  --resume -1 --save_results --ext sep


# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=4 --patch_size 48 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 2 --print_every 10 --test_every 100 --n_threads 4 \
# --save AIN_test_96x2_L1 --save_results --ext sep

# kill you can use this command to kill a process
# ps -aux| grep AIN| awk '{if($11=="python")print $2"\n"}'|xargs kill -9