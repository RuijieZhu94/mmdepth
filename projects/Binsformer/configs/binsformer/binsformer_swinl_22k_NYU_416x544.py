_base_ = [
    '../_base_/models/binsformer.py', '../../../../configs/_base_/datasets/nyu.py',
    '../../../../configs/_base_/default_runtime.py', 
]
custom_imports = dict(
    imports=['projects.Binsformer.decode_head'],
    allow_failed_imports=False)
crop_size = (416, 544)
data_preprocessor = dict(size=crop_size)

checkpoint_file = 'https://download.openmmlab.com/mmsegmentation/v0.5/pretrain/swin/swin_large_patch4_window7_224_22k_20220412-aeecf2aa.pth' # noqa
model = dict(
    data_preprocessor=data_preprocessor,
    backbone=dict(
        pretrain_img_size=224,
        embed_dims=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        init_cfg=dict(type='Pretrained', checkpoint=checkpoint_file)),
    decode_head=dict(
        type='BinsFormerDecodeHead',
        in_channels=[192, 384, 768,
                     1536],  # input channels of pixel_decoder modules
        num_queries=64,
    )
)

train_dataloader = dict(batch_size=6, num_workers=8)
val_dataloader = dict(batch_size=1, num_workers=4)
test_dataloader = val_dataloader