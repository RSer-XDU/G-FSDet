# Generalized Few-Shot Object Detection in Remote Sensing Images
This is the code for "Generalized Few-Shot Object Detection in Remote Sensing Images"

![](imgs/overall.png)

This code is based on [MMFewshow](https://github.com/open-mmlab/mmfewshot), you can see the mmfew for more detail about the instructions.



## Two-stage training framework


Following the original implementation, it consists of 3 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Reshape the bbox head of base model**:
   - create a new bbox head for all classes fine-tuning (base classes + novel classes) using provided script.
   - the weights of base class in new bbox head directly use the original one as initialization.
   - the weights of novel class in new bbox head use random initialization.

- **Step3: Few shot fine-tuning**:
   - use the base model from step2 as model initialization and further fine tune the bbox head with few shot datasets.


### An example of DIOR split1 1 shot setting with 2 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/ETF/dior/split1/tfa_r101_fpn_dior-split1_base-training.py 2

# step2: reshape the bbox head of base model for few shot fine-tuning
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/ETF_r101_fpn_voc-split1_base-training/latest.pth \
    --method randinit \
    --save-dir work_dirs/ETF_r101_fpn_voc-split1_base-training

# step3: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/ETF/dior/split1/ETF_r101_fpn_dior-split1_10shot-fine-tuning.py 8
```

**Note**:
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.

## Data preparation
We have provided  the few-shot annotations in 'data/few_shot_ann'. 

## Base training checkpoint on DIOR dataset
Split1: https://pan.baidu.com/s/11PcX-ywOiF3bPhFIcZKlUA     
        code：ouhu