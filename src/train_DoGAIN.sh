# 训练一个DoGAIN 96 
# CUDA_VISIBLE_DEVICES=3,4 python main.py --trainer trainer --template DoGAIN --n_GPUs 2 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-800/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 96 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 1000 --n_threads 16 \
# --save DoGAIN_96x2 --save_results --visdom --ext sep


# 这是一个AIN的baseline  在20张图上训练
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save AIN_20_baseline --save_results --visdom 
# --load AIN_20_baseline --resume -1
# --ext sep


# 2020年07月07日10:37:51 \sigma = 1/2, 1, 2, 4 三个尺度的模型
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template DoGAIN --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save DoGAIN_1_20 --save_results --visdom \
# --load DoGAIN_1_20 --resume -1
# --ext sep


# 2020年07月07日17:21:36 \sigma = 1/2, 1, 2, 4 两个尺度的模型
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template DoGAIN1 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save DoGAIN_2scales_20 --save_results --visdom \
# --load DoGAIN_2scales_20 --resume -1
# --ext sep

# 2020年07月07日22:41:53 两个尺度 增加归一化
# CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template DoGAIN2 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save DoGAIN_2scales_normal_20 --save_results --visdom \
# --load DoGAIN_2scales_normal_20 --resume -1
# --ext sep


# # 2020年07月08日11:32:36
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGAIN3 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save DoGAIN_2scales_constantnormal_20 --save_results --visdom \
# --load DoGAIN_2scales_constantnormal_20 --resume -1
# --ext sep


# 2020年07月08日14:35:53 
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGAIN4 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save DoGAIN_2scales_constantnormal2_20 --save_results --visdom \
# --load DoGAIN_2scales_constantnormal2_20 --resume -1 
# --ext sep


# 2020年07月08日15:14:47
# CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template DoGAIN5 --n_GPUs 1 --lr 0.0001 --loss 1*MSE \
# --data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
# --data_test Set5+Set14+Urban100+B100 \
# --epochs 400 --print_every 100 --test_every 200 --n_threads 16 \
# --save DoGAIN_2scales_constantnormal3_20 --save_results --visdom \
# --load DoGAIN_2scales_constantnormal3_20 --resume -1
# --ext sep



# 这是一个AIN2的baseline  在20张图上训练
# AIN2中把所有的Relu 都换成了 FRelu
# 为了训练这个模型，myupsampler中的relu 也换成了 frelu
# 要训练其他模型，这里要改回来
CUDA_VISIBLE_DEVICES=2 python main.py --trainer trainer --template AIN2 --n_GPUs 1 --lr 0.0001 --loss 1*L1 \
--data_train DIV2K --data_range 1-20/801-810 --n_colors 1 --scale 2 --batch_size=16 --patch_size 48 --direct_downsampling \
--data_test Set5+Set14+Urban100+B100 \
--epochs 200 --print_every 100 --test_every 200 --n_threads 16 \
--save AIN2_20_baseline_frelu --save_results --visdom \
--ext sep