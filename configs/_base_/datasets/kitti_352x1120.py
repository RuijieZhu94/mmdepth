# dataset settings
dataset_type = 'KITTIDataset'
data_root = 'data/kitti/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (352, 1120)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3),
    # dict(type='LoadKITTICamIntrinsic'),
    dict(type='KBCrop', depth=True),
    # dict(type='RandomRotate', prob=0.5, degree=2.5),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 1120)),
    dict(
        type='Albu',
        transforms=[
            dict(type='RandomBrightnessContrast'),
            dict(type='RandomGamma'),
            dict(type='HueSaturationValue'),
        ]),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'category_id')),
]


test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3),
    dict(type='KBCrop', depth=True),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction'
                   'category_id'))
]

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        data_prefix=dict(
            img_path='input', depth_map_path='gt_depth'),
        split='kitti_eigen_train.txt',
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        test_mode=True,
        data_prefix=dict(
            img_path='input', depth_map_path='gt_depth'),
        split='kitti_eigen_test.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=80.0,
    depth_scale_factor=256.0,
    crop_type='garg_crop')
test_evaluator = val_evaluator

