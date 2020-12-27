# 重新训练一个用于插值的RCAN 
# CUDA_VISIBLE_DEVICES=0 python main.py --trainer trainer --template RCAN \
# --data_train DIV2K --data_range 1-800/801-810 --data_test Set5+Set14+Urban100+B100 --direct_downsampling \
# --epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 --batch_size=16 \
# --save RCAN_DDX2_G10R20_400 --scale 2 --save_results --visdom --patch_size 96 \
# --resume -1 --load RCAN_DDX2_G10R20_400


# 训练一个resize的RCAN做参照
CUDA_VISIBLE_DEVICES=1 python main.py --trainer trainer --template RCAN \
--data_train DIV2K --data_range 1-800/801-810 --data_test Set5+Set14+Urban100+B100 \
--epochs 1000 --print_every 100 --test_every 1000 --n_threads 16 --batch_size=16 \
--save RCAN_X2_G10R20_400 --scale 2 --save_results --visdom --patch_size 96 \
--ext sep

