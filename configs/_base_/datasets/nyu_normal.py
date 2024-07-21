# dataset settings
dataset_type = 'NYUNormalDataset'
data_root = 'data/nyu_geonet'

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadDepthAnnotation', depth_rescale_factor=1e-3),
    dict(type='LoadNormalAnnotation', valid_mask=True),
    dict(type='RandomDepthMix', prob=0.25),
    dict(type='RandomFlip', prob=0.5),
    dict(type='RandomCrop', crop_size=(480, 480)),
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
    dict(type='Resize', scale=(2000, 480), keep_ratio=True),
    dict(dict(type='LoadDepthAnnotation', depth_rescale_factor=1e-3)),
    dict(type='LoadNormalAnnotation', valid_mask=True),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
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
            img_path='train/img', depth_map_path='train/depth',
            normal_map_path='train/normal', mask_map_path='train/mask'), 
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
            img_path='test/img', depth_map_path='test/depth',
            normal_map_path='test/normal', mask_map_path='test/mask'), 
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='NormalMetric',
    crop_type='nyu_crop')
test_evaluator = val_evaluator

