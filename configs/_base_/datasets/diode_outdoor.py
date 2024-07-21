# dataset settings
dataset_type = 'DIODEDataset'
data_root = 'data/DIODE'

# img size: 1024 x 768

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=480, keep_ratio=False),
    dict(dict(type='LoadDepthAnnotation', depth_rescale_factor=3.90625e-3)),
    dict(
        type='PackSegInputs',
        meta_keys=('img_path', 'depth_map_path', 'ori_shape', 'img_shape',
                   'pad_shape', 'scale_factor', 'flip', 'flip_direction',
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
            img_path='.', depth_map_path='.'),
        split='outdoor_test.txt',
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='DepthMetric',
    min_depth_eval=0.001,
    max_depth_eval=80.0,
    depth_scale_factor=256.0,
    crop_type='garg_crop')
test_evaluator = val_evaluator

