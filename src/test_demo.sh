# # x2
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGMSRNV5 --data_test Set5+Set14+B100+Urban100+Manga109 --save DoGMSRN_BIX2_output --pre_train ../experiment/DoGMSRN_BIX2_P48_v5_16layer_1000/model/model_latest.pt --scale 2 --save_results --patch_size 96 --ext sep  --reset --test_only

# # x2 ensemble
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGMSRNV5 --data_test Set5+Set14+B100+Urban100+Manga109 --save DoGMSRN_BIX2_output_ensemble --pre_train ../experiment/DoGMSRN_BIX2_P48_v5_16layer_1000/model/model_latest.pt --scale 2 --save_results --patch_size 96 --ext sep --reset --test_only --self_ensemble

# # x3 
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGMSRNV5 --data_test Set5+Set14+B100+Urban100+Manga109 --save DoGMSRN_BIX3_output --pre_train ../experiment/DoGMSRN_BIX3_P48_v5_16layer_1000/model/model_latest.pt --scale 3 --save_results --ext sep --reset --test_only

# #x3 ensemble
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGMSRNV5 --data_test Set5+Set14+B100+Urban100+Manga109 --save DoGMSRN_BIX3_output_ensemble --pre_train ../experiment/DoGMSRN_BIX3_P48_v5_16layer_1000/model/model_latest.pt --scale 3 --save_results --ext sep --reset --test_only --self_ensemble

# # x4 
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGMSRNV5 --data_test Set5+Set14+B100+Urban100+Manga109 --save DoGMSRN_BIX4_output --pre_train ../experiment/DoGMSRN_BIX4_P48_v5_16layer_1000/model/model_latest.pt --scale 4 --save_results --ext sep --reset --test_only

# #x4 ensemble
# CUDA_VISIBLE_DEVICES=4 python main.py --trainer trainer --template DoGMSRNV5 --data_test Set5+Set14+B100+Urban100+Manga109 --save DoGMSRN_BIX4_output_ensemble --pre_train ../experiment/DoGMSRN_BIX4_P48_v5_16layer_1000/model/model_latest.pt --scale 4 --save_results --ext sep --reset --test_only --self_ensemble


CUDA_VISIBLE_DEVICES=3 python main.py --trainer trainer --template RCAN --data_test Set5 --save RCAN_BIX2_output --pre_train ../experiment/models_ECCV2018RCAN/RCAN_BIX2.pt --scale 2 --save_results --patch_size 96 --ext sep  --reset --test_only

