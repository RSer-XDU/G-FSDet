# Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild <a href="https://arxiv.org/abs/2007.12107"> (ECCV'2020)</a>

## Abstract

<!-- [ABSTRACT] -->
Detecting objects and estimating their viewpoint in images are key tasks of 3D scene understanding.
Recent approaches have achieved excellent results on very large benchmarks for object detection and view-point estimation.
However, performances are still lagging behind for novel object categories with few samples.
In this paper, we tackle the problems of few-shot object detection and few-shot viewpoint estimation.
We propose a meta-learning framework that can be applied to both tasks, possibly including 3D data.
Our models improve the results on objects of novel classes by leveraging on rich feature information originating from base classes with many samples. A simple joint
feature embedding module is proposed to make the most of this feature sharing.
Despite its simplicity, our method outperforms state-of-the-art methods by a large margin on a range of datasets, including
PASCAL VOC and MS COCO for few-shot object detection, and Pascal3D+ and ObjectNet3D for few-shot viewpoint estimation.
And for the first time, we tackle the combination of both few-shot tasks, on ObjectNet3D, showing promising results.
Our code and data are available at http://imagine.enpc.fr/~xiaoy/FSDetView/.


<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142845154-a50d8902-1b7a-4c7e-9b36-0848ff080187.png" width="80%"/>
</div>


## Citation

<!-- [ALGORITHM] -->

```bibtex
@inproceedings{xiao2020fsdetview,
    title={Few-Shot Object Detection and Viewpoint Estimation for Objects in the Wild},
    author={Yang Xiao and Renaud Marlet},
    booktitle={European Conference on Computer Vision (ECCV)},
    year={2020}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.

## How to reproduce FSDetView

Following the original implementation, it consists of 2 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Few shot fine-tuning**:
   - use the base model from step1 as model initialization and further fine tune the model with few shot datasets.


### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_base-training.py 8

# step2: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/fsdetview/voc/split1/fsdetview_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8
```

**Note**:
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.




