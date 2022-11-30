CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/GFSDet/dior/split1/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_10shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/GFSDet/dior/split1/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_3shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/GFSDet/dior/split1/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_5shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/GFSDet/dior/split1/power4_0.025_weight_0.5_alpha_tfa_r101_fpn_dior-split1_20shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/dis_loss/dior/split1/power4_dis_tfa_r101_fpn_dior-split1_3shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/dis_loss/dior/split1/power4_dis_tfa_r101_fpn_dior-split1_5shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/dis_loss/dior/split1/power4_dis_tfa_r101_fpn_dior-split1_10shot-fine-tuning.py 2
CUDA_VISIBLE_DEVICES=0,2 bash tools/detection/dist_train.sh configs/detection/dis_loss/dior/split1/power4_dis_tfa_r101_fpn_dior-split1_20shot-fine-tuning.py 2


