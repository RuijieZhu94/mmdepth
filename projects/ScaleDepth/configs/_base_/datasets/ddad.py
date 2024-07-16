# dataset settings
dataset_type = 'DDADDataset'
data_root = 'data/ddad/ddad_results'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# img size = (1936, 1216)

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(1120, 352), keep_ratio=False),     
    dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3),    
    # dict(type='KBCrop', depth=True),
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
            img_path='input', depth_map_path='gt_depth'),
        split='camera01.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=80.0,
    depth_scale_factor=256.0,
    depth_clamp=True,
    crop_type='garg_crop')
test_evaluator = val_evaluator

