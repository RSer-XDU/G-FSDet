_base_ = [
    '../../../_base_/datasets/fine_tune_based/few_shot_dior.py',
    '../../../_base_/schedules/schedule.py', '../../tfa_r101_fpn.py',
    '../../../_base_/default_runtime.py'
]
# classes splits are predefined in FewShotVOCDataset
# FewShotVOCDefaultDataset predefine ann_cfg for model reproducibility.
data = dict(
    train=dict(
        type='FewShotDIORDefaultDataset',
        ann_cfg=[dict(method='TFA', setting='SPLIT1_3SHOT')],
        num_novel_shots=3,
        num_base_shots=3,
        classes='ALL_CLASSES_SPLIT1'),
    val=dict(classes='ALL_CLASSES_SPLIT1'),
    test=dict(classes='ALL_CLASSES_SPLIT1'))
evaluation = dict(
    interval=6000,
    class_splits=['BASE_CLASSES_SPLIT1', 'NOVEL_CLASSES_SPLIT1'])
checkpoint_config = dict(interval=2000)
optimizer = dict(lr=0.001)
optimizer_config=dict(_delete_=True, grad_clip=dict(max_norm=20, norm_type=2))
lr_config = dict(
    warmup_iters=100, step=[
        6000,
    ])
runner = dict(max_iters=6000)
model = dict(
    pretrained='open-mmlab://detectron2/resnet101_caffe',
    backbone=dict(depth=101),
    frozen_parameters=[
        'backbone', 'neck', 'roi_head.bbox_head.base_shared_fcs',],

    roi_head=dict(
        bbox_head=dict(
            type='DisKDBBoxHead',
            num_classes=20,
            loss_kd_weight= 0.025,
            base_alpha=0.5,
            loss_bbox=dict(loss_weight=2.0),
            loss_cls=dict(loss_weight=1.0),
            dis_loss=dict(
                type='DisLoss', num_classes=20, shot=3, 
                loss_base_margin_weight=1.0,
                loss_novel_margin_weight=1.0,
                loss_neg_margin_weight=1.0,
                power_weight=4.0),
            base_cpt = 'work_dirs/DIOR/TFA/tfa_r101_fpn_dior-split1_base-training/base_model_random_init_bbox_head.pth',
            init_cfg=[
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='base_shared_fcs')),
                dict(
                    type='Caffe2Xavier',
                    override=dict(type='Caffe2Xavier', name='novel_shared_fcs')),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_cls', std=0.01)),
                dict(
                    type='Normal',
                    override=dict(type='Normal', name='fc_reg', std=0.001))
            ]
            )))
# base model needs to be initialized with following script:
#   tools/detection/misc/initialize_bbox_head.py
# please refer to configs/detection/tfa/README.md for more details.

load_from = ('work_dirs/DIOR/TFA/tfa_r101_fpn_dior-split1_base-training/'
             'base_model_random_init_bbox_head.pth')
