_base_ = [
    '../_base_/models/scaledepth.py', '../../../../configs/_base_/datasets/nyu_480x480.py',
    '../../../../configs/_base_/default_runtime.py', 
]
custom_imports = dict(
    imports=['mmpretrain.models', 'projects.ScaleDepth.decode_head'],
    allow_failed_imports=False)
crop_size = (480, 480)
data_preprocessor = dict(size=crop_size)

pretrained = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-large_3rdparty_in21k_20220301-e6e0ea0a.pth'  # noqa

model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        type='mmpretrain.ConvNeXt',
        arch='large',
        out_indices=[0, 1, 2, 3],
        drop_path_rate=0.4,
        layer_scale_init_value=1.0,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=pretrained,
            prefix='backbone.')),
    decode_head=dict(
        type="ScaleDepthDecodeHead",
        in_channels=[192, 384, 768, 1536],  # input channels of pixel_decoder modules
        num_queries=64,
        num_classes=1,
        with_classify=True,
        with_scale=True,
        class_embed_path='projects/ScaleDepth/pretrained_weights/nyu_class_embeddings_convnext_large_d_320.pth'
    ),
    test_cfg=dict(mode='slide_flip', crop_size=crop_size, stride=(160, 160))
)

train_dataloader = dict(batch_size=6, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader

default_hooks = dict(
    checkpoint=dict(save_best='rmse', rule='less', max_keep_ckpts=1))

find_unused_parameters = True
