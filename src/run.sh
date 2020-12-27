# RCAN
# 这两个实验用于重现RCAN的结果
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template RCAN --data_train DIV2K --data_range 1-800/801-810 --batch_size=16 --data_test Set5+Set14+Urban100+B100 --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 --save RCAN_BIX2_G10R20_1000 --scale 2 --save_results --visdom --patch_size 96 --ext sep

# CUDA_VISIBLE_DEVICES=0,3 python main.py --trainer trainer --template RCAN --n_GPUs=2 \
# --lr 1e-4 --decay 20 --epochs 100 \
# --data_train DIV2K --data_range 1-800/801-810 --batch_size=16 --data_test Set5+Set14+Urban100+B100 --patch_size 136 \
# --print_every 10 --test_every 1000 --n_threads 16 --save RCAN_BIX2_G10R20_1000_136 \
# --scale 2 --save_results --visdom --ext sep \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt

# CUDA_VISIBLE_DEVICES=0,1,2 python main.py --trainer trainer --template RCAN --n_GPUs=3 \
# --lr 1e-4 --decay 20 --epochs 100 \
# --data_train DIV2K --data_range 1-800/801-810 --batch_size=16 --data_test Set5+Set14+Urban100+B100 --patch_size 192 \
# --print_every 10 --test_every 1000 --n_threads 16 --save RCAN_BIX2_G10R20_1000_192 \
# --scale 2 --save_results --visdom --ext sep \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt


# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_train DIV2K --data_range 1-800/801-810 --batch_size=16 --data_test Set5+Set14+Urban100 --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 --save RCAN_BIX2_G10R20P48_1000_2 --scale 2 --save_results --visdom --patch_size 96 --ext sep

# kill 
# ps -aux| grep sig0.7| awk '{if($11=="python")print $2"\n"}'|xargs kill -9






# 用RCAN重建带噪声的图像
# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo ../test/Set5gaussian --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set5gaussian --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo ../test/Set5saltpepper --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set5saltpepper --save_results

# CUDA_VISIBLE_DEVICES=2 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set5_LR_bicubic_X2gaussian --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set5_bicubicx2_gaussian --save_results

# CUDA_VISIBLE_DEVICES=2 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set5_LR_bicubic_X2saltpepper --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set5_bicubicx2_saltpepper --save_results

# CUDA_VISIBLE_DEVICES=3 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/oct --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Oct --save_results

# CUDA_VISIBLE_DEVICES=4 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/low-light/small --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save lowlight --save_results

CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set14 --scale 2 --n_colors 3 \
--pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set14x2 --save_results

CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo ../experiment/Set14x2/results-Demo --scale 2 --n_colors 3 \
--pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set14x4 --save_results

# CUDA_VISIBLE_DEVICES=0 python main.py --template RCAN --data_test Demo --dir_demo ../experiment/Set14x4/results-Demo --scale 2 --n_colors 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set14x8 --save_results

# CUDA_VISIBLE_DEVICES=4 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set14_LR_bicubic_X2gaussian --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set14_LR_bicubic_X2gaussian --save_results

# CUDA_VISIBLE_DEVICES=4 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set14saltpepper --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save Set14saltpepper --save_results


# 下面的是用RCAN 重建 直接下采样的图
# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set5x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Set5_INTP_x2 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set15x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Set15_INTP_x2 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set18x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Set18_INTP_x2 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Urban12x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Urban12_INTP_x2 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Urban100x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Urban100_INTP_x2 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/BSDS100x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_BSDS100_INTP_x2 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Manga109x2 --scale 2 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Manga109_INTP_x2 --save_results

# # x3 
# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set5x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Set5_INTP_x3 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set15x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Set15_INTP_x3 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Set18x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Set18_INTP_x3 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Urban12x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Urban12_INTP_x3 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Urban100x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Urban100_INTP_x3 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/BSDS100x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_BSDS100_INTP_x3 --save_results

# CUDA_VISIBLE_DEVICES=1 python main.py --template RCAN --data_test Demo --dir_demo /data/jjh_backup/1_3/testset/Manga109x3 --scale 3 \
# --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --test_only --save RCAN_Manga109_INTP_x3 --save_results