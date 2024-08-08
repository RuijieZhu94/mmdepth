# dataset settings
nyu_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation', depth_rescale_factor=1e-3),
    # dict(type='RandomDepthMix', prob=0.25), # for mix-dataset training
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 512)),
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

kitti_train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3),
    dict(type='KBCrop', depth=True),
    # dict(type='Resize', scale=(384, 2000), keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(352, 512)),
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

nyu_test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(2000, 480), keep_ratio=True),
    dict(dict(type='LoadDepthAnnotation', depth_rescale_factor=1e-3)),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'category_id'))
]

kitti_test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2000, 480), keep_ratio=True),
    dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3),
    dict(type='KBCrop', depth=True),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction'
                   'category_id'))
]

dataset_nyu_train = dict(
    type='NYUDataset',
    data_root='data/nyu',
    data_prefix=dict(
        img_path='images/train', depth_map_path='annotations/train'),
    pipeline=nyu_train_pipeline)

dataset_kitti_train = dict(
    type='KITTIDataset',
    data_root='data/kitti/',
    data_prefix=dict(
        img_path='input', depth_map_path='gt_depth'),
    split='kitti_eigen_train.txt',
    pipeline=kitti_train_pipeline)

nyu_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='NYUDataset',
        data_root='data/nyu',
        test_mode=True,
        data_prefix=dict(
            img_path='images/test', depth_map_path='annotations/test'),
        pipeline=nyu_test_pipeline))

kitti_val_dataloader = dict(
    batch_size=1,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='KITTIDataset',
        data_root='data/kitti/',
        test_mode=True,
        data_prefix=dict(
            img_path='input', depth_map_path='gt_depth'),
        split='kitti_eigen_test.txt',
        pipeline=kitti_test_pipeline))

nyu_val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=10.0,
    depth_scale_factor=1000.0,
    depth_clamp=True,
    crop_type='nyu_crop')

kitti_val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=80.0,
    depth_scale_factor=256.0,
    depth_clamp=True,
    crop_type='garg_crop')

train_dataloader = dict(
    batch_size=8,
    num_workers=8,
    persistent_workers=True,
    sampler=dict(type='InfiniteSampler', shuffle=True),
    dataset=dict(
        type='ConcatDataset',
        datasets=[dataset_nyu_train, dataset_kitti_train]))

# val_dataloader = kitti_val_dataloader
val_dataloader = nyu_val_dataloader
test_dataloader = val_dataloader

# val_evaluator = kitti_val_evaluator
val_evaluator = nyu_val_evaluator
test_evaluator = val_evaluator

