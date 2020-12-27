# 训练一个AIN 96 
CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
--data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
--data_test Set5+Set14+Urban100+B100 \
--epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
--load AIN_96x2 --save AIN_96x2 --save_results --visdom --ext sep --onlydraw

# 96 * 5 / 4 = 120
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 120 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_120x2 --save_results --visdom --ext sep

# # 96 * 6 / 4 = 144
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 144 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_144x2 --save_results --visdom --ext sep

# # 96 * sqrt(2) = 136
# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 136 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_136x2 --save_results --visdom --ext sep

# 96 * 7 / 4 = 168
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 168 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_168x2 --save_results --visdom --ext sep

# 96 * 8 / 4 = 192
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 192 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_192x2 --save_results --visdom --ext sep


# 96 * 8 / 4 = 272 # 已经完全超出GPU的显存了
# CUDA_VISIBLE_DEVICES=1,2,3,4 python main.py --trainer trainer --template AIN --n_GPUs 4 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 272 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN_272x2 --save_results --visdom --ext sep



# kill 
# ps -aux| grep sig0.7| awk '{if($11=="python")print $2"\n"}'|xargs kill -9


# 训练一个AIN 96 
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template AIN --n_resblocks 1 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN1_96x2 --save_results --visdom --ext sep

# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template AIN --n_resblocks 2 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN2_96x2 --save_results --visdom --ext sep

# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN --n_resblocks 3 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN3_96x2 --save_results --visdom --ext sep

# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template AIN --n_resblocks 4 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN4_96x2 --save_results --visdom


# CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template AIN --n_resblocks 5 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN5_96x2 --save_results --visdom --ext sep

# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template AIN --n_resblocks 6 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN6_96x2 --save_results --visdom --ext sep

# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template AIN --n_resblocks 8 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN8_96x2 --save_results --visdom --ext sep


# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN0 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN0_96x2 --save_results --visdom --ext sep


# AIN1 has inception + channel-wise
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN1 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save AIN1_96x2 --save_results --visdom \
# --resume -1 --load AIN1_96x2