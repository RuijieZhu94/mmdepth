# dataset settings
dataset_type = 'SUNRGBDDataset'
data_root = 'data/sunrgbd/'

test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='Resize', scale=(2000, 480), keep_ratio=True),
    dict(dict(type='LoadDepthAnnotation', depthinpaint=True)),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
                   'category_id', 'focal_length'))
]

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
    max_depth_eval=8.0,
    depth_scale_factor=1000.0,
    depth_clamp=True,
    gt_depth_clamp=True,
    focal_length_rescale=True,
    crop_type='nyu_crop')
test_evaluator = val_evaluator

