_base_ = [
    '../_base_/models/scaledepth.py', '../../../../configs/_base_/datasets/kitti_352x1120.py',
    '../../../../configs/_base_/default_runtime.py', 
]
custom_imports = dict(
    imports=['projects.ScaleDepth.backbone', 'projects.ScaleDepth.neck', 'projects.ScaleDepth.decode_head'],
    allow_failed_imports=False)
crop_size = (352, 1120)
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[0, 0, 0],
    std=[1, 1, 1],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0,
    size=crop_size)

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type="CLIP",
        embed_dims = [192, 384, 768, 1536],
        clip_scales = ['res2', 'res3', 'res4', 'res5'],
        clip_model_name = 'convnext_large_d_320',
        clip_model_pretrain = 'laion2b_s29b_b131k_ft_soup',
        finetune=True,
        norm_cfg = dict(type='SyncBN', requires_grad=True),
        act_cfg = dict(type='LeakyReLU', inplace=True),
    ),      
    decode_head=dict(
        type="ScaleDepthDecodeHead",
        in_channels=[192, 384, 768, 1536],  # input channels of pixel_decoder modules
        num_queries=64,
        num_classes=1,
        with_classify=False,
        with_scale=True,
        min_depth=1e-3,
        max_depth=None,
    ),
    # test_cfg=dict(mode='slide_flip', crop_size=crop_size, stride=(160, 160))
)

train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(save_best='abs_rel', rule='less', max_keep_ckpts=1))

find_unused_parameters = True
# train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)