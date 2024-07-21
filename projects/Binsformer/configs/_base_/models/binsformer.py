# model settings
data_preprocessor = dict(
    type='SegDataPreProcessor',
    mean=[127.5, 127.5, 127.5],
    std=[127.5, 127.5, 127.5],
    bgr_to_rgb=True,
    pad_val=0,
    seg_pad_val=0)

norm_cfg = dict(type='SyncBN', requires_grad=True)
backbone_norm_cfg = dict(type='LN', requires_grad=True)

model = dict(
    type='DepthEstimator',
    data_preprocessor=data_preprocessor,
    decode_head=dict(
        type='BinsFormerDecodeHead',
        in_channels=[96, 192, 384,
                     768],  # input channels of pixel_decoder modules
        feat_channels=256,
        in_index=[0, 1, 2, 3],
        num_classes=1,
        out_channels=256,
        num_queries=64,
        pixel_decoder=dict(
            type='mmdet.PixelDecoder',
            norm_cfg=dict(type='GN', num_groups=32),
            act_cfg=dict(type='ReLU')),
        enforce_decoder_input_project=False,
        positional_encoding=dict(  # SinePositionalEncoding
            num_feats=128, normalize=True),
        transformer_decoder=dict(  # DetrTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # DetrTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    attn_drop=0.1,
                    proj_drop=0.1,
                    dropout_layer=None,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    act_cfg=dict(type='ReLU', inplace=True),
                    ffn_drop=0.1,
                    dropout_layer=None,
                    add_identity=True)),
            init_cfg=None)),
    train_cfg=dict(
        aux_loss = True,
        aux_index = [2, 5],
        aux_weight = [1/4, 1/2]),
    test_cfg=dict(mode='whole')
)
# optimizer
optimizer = dict(
    type='AdamW', lr=0.0001, betas=(0.9, 0.999), weight_decay=0.0001)
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=optimizer,
    clip_grad=dict(max_norm=0.01, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'backbone': dict(lr_mult=0.1),
    }))
# learning policy
param_scheduler = [
    dict(
        type='PolyLR',
        eta_min=0,
        power=0.9,
        begin=0,
        end=40000,
        by_epoch=False)
]

# training schedule for 40k
train_cfg = dict(type='IterBasedTrainLoop', max_iters=40000, val_interval=1000)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')
default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=50, log_metric_by_epoch=False),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(type='CheckpointHook', by_epoch=False, interval=2000),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='SegVisualizationHook'))

find_unused_parameters = True