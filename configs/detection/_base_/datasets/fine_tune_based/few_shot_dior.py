# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=(800, 800),
        keep_ratio=True,
        multiscale_mode='value'),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
# classes splits are predefined in FewShotDIORDataset
# FewShotDIORDefaultDataset predefine ann_cfg for model reproducibility.
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(800, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
# classes splits are predefined in FewShotDIORDataset
data_root = 'data/DIOR/'
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type='FewShotDIORDataset',
        save_dataset=True,
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'DIOR2017/ImageSets/Main/trainval.txt'),

        ],
        img_prefix=data_root,
        num_novel_shots=None,
        num_base_shots=None,
        pipeline=train_pipeline,
        classes=None,
        use_difficult=False,
        instance_wise=False),
    val=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'DIOR2017/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        classes=None),
    test=dict(
        type='FewShotDIORDataset',
        ann_cfg=[
            dict(
                type='ann_file',
                ann_file=data_root + 'DIOR2017/ImageSets/Main/test.txt')
        ],
        img_prefix=data_root,
        pipeline=test_pipeline,
        test_mode=True,
        classes=None))
evaluation = dict(interval=2000, metric='mAP', class_splits=None)
