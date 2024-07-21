# dataset settings
dataset_type = 'KITTIBenchmarkDataset'
data_root = 'data/kitti/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
crop_size= (352, 1120)
test_pipeline = [
    dict(type='LoadImageFromFile'),
    # dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3),
    # dict(type='KBCrop', depth=False),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction'
                   'category_id'))
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
            img_path='benchmark_test'),
        split='benchmark_test.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=80.0,
    depth_scale_factor=256.0,
    depth_clamp=True,
    format_only=True,
    crop_type='garg_crop')
test_evaluator = val_evaluator

