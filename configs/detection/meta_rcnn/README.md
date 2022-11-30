# Meta R-CNN: Towards General Solver for Instance-level Low-shot Learning <a href="https://arxiv.org/pdf/1909.13032.pdf"> (ICCV'2019)</a>

## Abstract

<!-- [ABSTRACT] -->

Resembling the rapid learning capability of human, low-shot learning empowers vision systems to understand new concepts by training with few samples.
Leading approaches derived from meta-learning on images with a single visual object.
Obfuscated by a complex background and multiple objects in one image, they are hard to promote the research of low-shot object detection/segmentation.
In this work, we present a flexible and general methodology to achieve these tasks.
Our work extends Faster /Mask R-CNN by proposing meta-learning over RoI (Region-of-Interest) features instead of a full image feature.
This simple spirit disentangles multi-object information merged with the background, without bells and whistles, enabling Faster / Mask R-CNN turn into a meta-learner to achieve the tasks.
Specifically, we introduce a Predictor-head Remodeling Network (PRN) that shares its main backbone with Faster / Mask R-CNN.
PRN receives images containing low-shot objects with their bounding boxes or masks to infer their class attentive vectors.
The vectors take channel-wise soft-attention on RoI features, remodeling those R-CNN predictor heads to detect or segment the objects consistent with the classes these vectors represent.
In our experiments, Meta R-CNN yields the new state of the art in low-shot object detection and improves low-shot object segmentation by Mask R-CNN.
Code: https://yanxp.github.io/metarcnn.html.


<!-- [IMAGE] -->
<div align=center>
<img src="https://user-images.githubusercontent.com/15669896/142843770-6390a0b2-f40a-4731-ad4d-b6ab4c8268b8.png" width="80%"/>
</div>

## Citation

<!-- [ALGORITHM] -->
```bibtex
@inproceedings{yan2019meta,
    title={Meta r-cnn: Towards general solver for instance-level low-shot learning},
    author={Yan, Xiaopeng and Chen, Ziliang and Xu, Anni and Wang, Xiaoxi and Liang, Xiaodan and Lin, Liang},
    booktitle={Proceedings of the IEEE International Conference on Computer Vision},
    year={2019}
}
```

**Note**: ALL the reported results use the data split released from [TFA](https://github.com/ucbdrive/few-shot-object-detection/blob/main/datasets/README.md) official repo.
Currently, each setting is only evaluated with one fixed few shot dataset.
Please refer to [DATA Preparation](https://github.com/open-mmlab/mmfewshot/tree/main/tools/data/detection) to get more details about the dataset and data preparation.

## How to reproduce Meta RCNN

Following the original implementation, it consists of 2 steps:
- **Step1: Base training**
   - use all the images and annotations of base classes to train a base model.

- **Step2: Few shot fine-tuning**:
   - use the base model from step1 as model initialization and further fine tune the model with few shot datasets.


### An example of VOC split1 1 shot setting with 8 gpus

```bash
# step1: base training for voc split1
bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_base-training.py 8

# step2: few shot fine-tuning
bash ./tools/detection/dist_train.sh \
    configs/detection/meta_rcnn/voc/split1/meta-rcnn_r101_c4_8xb4_voc-split1_1shot-fine-tuning.py 8
```

**Note**:
- The default output path of the reshaped base model in step2 is set to `work_dirs/{BASE TRAINING CONFIG}/base_model_random_init_bbox_head.pth`.
  When the model is saved to different path, please update the argument `load_from` in step3 few shot fine-tune configs instead
  of using `resume_from`.
- To use pre-trained checkpoint, please set the `load_from` to the downloaded checkpoint path.
