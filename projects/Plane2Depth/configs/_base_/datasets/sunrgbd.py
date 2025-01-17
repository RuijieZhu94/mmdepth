# dataset settings
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'
# train_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(type='LoadDepthAnnotation', depth_rescale_factor=1e-3),
#     dict(type='RandomDepthMix', prob=0.25),
#     dict(type='RandomFlip', prob=0.5),
#     dict(type='RandomCrop', crop_size=(416, 544)),
#     dict(
#         type='Albu',
#         transforms=[
#             dict(type='RandomBrightnessContrast'),
#             dict(type='RandomGamma'),
#             dict(type='HueSaturationValue'),
#         ]),
#     dict(
#         type='PackSegInputs',
#         meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
#                    'pad_shape', 'scale_factor', 'flip', 'flip_direction',
#                    'category_id')),
# ]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(640, 480), keep_ratio=False),
    dict(dict(type='LoadDepthAnnotation', depthinpaint=True)),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'category_id', 'focal_length','intrinsics'))
]

# train_dataloader = dict(
#     batch_size=8,
#     num_workers=8,
#     persistent_workers=True,
#     sampler=dict(type='InfiniteSampler', shuffle=True),
#     dataset=dict(
#         type=dataset_type,
#         data_root=data_root,
#         data_prefix=dict(
#             img_path='', depth_map_path=''),
#         pipeline=train_pipeline))

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
            img_path='', depth_map_path=''),
        split='SUNRGBD_val_splits.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=10.0,
    depth_scale_factor=1000.0,
    depth_clamp=True,
    gt_depth_clamp=True,
    focal_length_rescale=True,
    crop_type='nyu_crop')
test_evaluator = val_evaluator

