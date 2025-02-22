# train an AIN with patch size set to 96 
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_96x2 --save_results --visdom --ext sep


## test AIN 96
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template AIN \
# --n_colors 1 --scale 2 --direct_downsampling --data_test mySet5+myUrban12+myUrban100 --load AIN_96x2 --resume -1 --test_only --save_results


# # 96 * sqrt(2) = 136
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 136 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_136x2 --save_results --visdom --ext sep

# 96 * 2 = 192
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 192 --direct_downsampling \
# --data_test mySet5+myUrban12 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_192x2 --save_results --visdom --ext sep

# You can use this command to kill a process
# ps -aux| grep AIN| awk '{if($11=="python")print $2"\n"}'|xargs kill -9


# AIN2 96 
CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template AIN2 --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
--data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
--data_test mySet5+myUrban12 \
--epochs 400 --print_every 10 --test_every 10 --n_threads 16 \
--save AIN2_96x2_L1 --save_results --visdom --ext sep