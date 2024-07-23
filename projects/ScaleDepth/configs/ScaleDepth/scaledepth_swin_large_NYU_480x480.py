_base_ = [
    '../_base_/models/scaledepth.py', '../../../../configs/_base_/datasets/nyu_480x480.py',
    '../../../../configs/_base_/default_runtime.py', 
]
custom_imports = dict(
    imports=['mmpretrain.models', 'projects.ScaleDepth.decode_head'],
    allow_failed_imports=False)
crop_size = (480, 480)
data_preprocessor = dict(size=crop_size)

pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window12_384_22k_20220412-6580f57d.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='SwinTransformer',
        pretrain_img_size=384,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=12,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        frozen_stages=-1,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    decode_head=dict(
        type="ScaleDepthDecodeHead",
        in_channels=[192, 384, 768, 1536],  # input channels of pixel_decoder modules
        num_queries=64,
        num_classes=1,
        with_classify=True,
        with_scale=True,
        class_embed_path='./nyu_class_embeddings_convnext_large_d_320.pth'
    ),
    # test_cfg=dict(mode='slide_flip', crop_size=crop_size, stride=(160, 160))
)

train_dataloader = dict(batch_size=6, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(save_best='rmse', rule='less', max_keep_ckpts=1))

find_unused_parameters = True
