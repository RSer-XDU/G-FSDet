# Frustratingly Simple Few-Shot Object Detection <a href="https://arxiv.org/abs/2003.06957">(ICML'2020)</a>


## Abstract

<!-- [ABSTRACT] -->

Detecting rare objects from a few examples is an emerging problem.
Prior works show meta-learning is a promising approach.
But, fine-tuning techniques have drawn scant attention.
We find that fine-tuning only the last layer of existing detectors on rare classes is crucial to the few-shot object detection task.
Such a simple approach outperforms the meta-learning methods by roughly 2~20 points on current benchmarks and sometimes even doubles the accuracy of the prior methods.
However, the high variance in the few samples often leads to the unreliability of existing benchmarks.
We revise the evaluation protocols by sampling multiple groups of training examples to obtain stable comparisons and build new benchmarks based on three datasets: PASCAL VOC, COCO and LVIS.
Again, our fine-tuning approach establishes a new state of the art on the revised benchmarks.
The code as well as the pretrained models are available at https://github.com/ucbdrive/few-shot-object-detection.

<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142841882-4266e4a6-b93f-44d1-9754-72be2473d589.png" width="80%"/>
</div>



## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{wang2020few,
    title={Frustratingly Simple Few-Shot Object Detection},
    author={Wang, Xin and Huang, Thomas E. and  Darrell, Trevor and Gonzalez, Joseph E and Yu, Fisher}
    booktitle={International Conference on Machine Learning (ICML)},
    year={2020}
}
```



**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.


## How to reproduce TFA


Following the original implementation, it consists of 3 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Reshape the bbox head of base model**:
   - create a new bbox head for all classes fine-tuning (base classes + novel classes) using provided script.
   - the weights of base class in new bbox head directly use the original one as initialization.
   - the weights of novel class in new bbox head use random initialization.

- **Step3: Few shot fine-tuning**:
   - use the base model from step2 as model initialization and further fine tune the bbox head with few shot datasets.


### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_base-training.py 8

# step2: reshape the bbox head of base model for few shot fine-tuning
python -m tools.detection.misc.initialize_bbox_head \
    --src1 work_dirs/tfa_r101_fpn_voc-split1_base-training/latest.pth \
    --method randinit \
    --save-dir work_dirs/tfa_r101_fpn_voc-split1_base-training

# step3: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/tfa/voc/split1/tfa_r101_fpn_voc-split1_1shot-fine-tuning.py 8
```

**Note**:
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.




