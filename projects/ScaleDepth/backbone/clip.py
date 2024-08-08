import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule
from mmdepth.registry import MODELS
# from ..utils import FeatureExtractor
# from mmengine.model import BaseModule
from mmengine.runner import CheckpointLoader, load_checkpoint

@MODELS.register_module()
class CLIP(nn.Module):
    """Feature2Pyramid.

    A neck structure connect ViT backbone and decoder_heads.

    Args:
        embed_dims (int): Embedding dimension.
        rescales (list[float]): Different sampling multiples were
            used to obtain pyramid features. Default: [4, 2, 1, 0.5].
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='SyncBN', requires_grad=True).
    """

    def __init__(self,
                 embed_dims=[192, 384, 768, 1536],
                 clip_scales=[3, 5, 7, 9],
                 clip_model_name: str = 'ViT-B-16',
                 clip_model_pretrain: str = 'openai',
                 clip_channel: int = 768,
                 prompt_depth: int = 0,
                 prompt_length: int = 0,
                 finetune: bool = False,
                 finetune_bias: bool = False,
                 class_embed_path: str = 'https://download.openmmlab.com/mmsegmentation/'
                 'v0.5/vpd/nyu_class_embeddings.pth',
                 class_embed_select: bool = False,
                 norm_cfg=dict(type='SyncBN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', inplace=True)):
        super().__init__()
        self.embed_dims = embed_dims
        self.clip_scales = clip_scales
        self.finetune = finetune
        self.finetune_bias = finetune_bias
        self.class_embed_select = class_embed_select
        # clip model
        import open_clip
        self.clip_model, _, clip_preprocess = open_clip.create_model_and_transforms(clip_model_name, pretrained=clip_model_pretrain)
        # self.clip_model = clip_model.float()
        # self.clip_tokenizer = open_clip.get_tokenizer('ViT-B-16')
        clip_pixel_mean, clip_pixel_std = (
            [m * 255. for m in clip_preprocess.transforms[-1].mean],
            [s * 255. for s in clip_preprocess.transforms[-1].std],
        )
        self.clip_resolution = clip_preprocess.transforms[1].size # (384, 384)

        # normalization parameters
        self.register_buffer('clip_pixel_mean',
                             torch.Tensor(clip_pixel_mean).view(1, -1, 1, 1),
                             False)
        self.register_buffer('clip_pixel_std',
                             torch.Tensor(clip_pixel_std).view(1, -1, 1, 1),
                             False)

        model_name = clip_model_name.lower()
        if 'convnext_' in model_name:
            self.model_type = 'convnext'
            if '_base' in model_name:
                self.output_channels = [128, 128, 256, 512, 1024]
            elif '_large' in model_name:
                self.output_channels = [192, 192, 384, 768, 1536]
            elif '_xxlarge' in model_name:
                self.output_channels = [384, 384, 768, 1536, 3072]
        
        elif 'rn' in model_name:
            self.model_type = 'resnet'
            if model_name.replace('-quickgelu', '') in ['rn50', 'rn101']:
                self.output_channels = [64, 256, 512, 1024, 2048]
            elif model_name == 'rn50x4':
                self.output_channels = [80, 320, 640, 1280, 2560]
            elif model_name == 'rn50x16':
                self.output_channels = [96, 384, 768, 1536, 3072]
            elif model_name == 'rn50x64':
                self.output_channels = [128, 512, 1024, 2048, 4096]
        # elif 'vit' in model_name:
        #     self.clip_visual_extractor = FeatureExtractor(
        #         self.clip_model.visual,
        #         last_layer_idx=9,
        #         frozen_exclude=['positional_embedding'],
        #     )
        if not self.finetune:
            self.freeze_everything()
        if self.finetune_bias:
            self.freeze_except_bias()

        if self.class_embed_select:
            class_embeddings = CheckpointLoader.load_checkpoint(class_embed_path)
            self.register_buffer('class_embeddings', class_embeddings)
        # feature fusion layer
        # self.feature_fusion_layer = nn.ModuleList()
        # for embed_dim in self.embed_dims:
        #     self.feature_fusion_layer.append(
        #         ConvModule(clip_channel,
        #                     embed_dim,
        #                     kernel_size=1,
        #                     norm_cfg=norm_cfg,
        #                     act_cfg=act_cfg)
        #     )

    def freeze_everything(self):
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def freeze_except_bias(self):
        for key, value in self.clip_model.named_parameters():
            if not 'bias' in key:
                value.requires_grad = False
                
    def clip_preprocess(self, inputs):
        """Input normalization for clip model and feature extractor
        respectively.

        Args:
            inputs: batched input images.
        """
        # clip images
        batched_clip = (inputs - self.clip_pixel_mean) / self.clip_pixel_std
        # batched_clip = F.interpolate(
        #     batched_clip,
        #     size=self.clip_resolution,
        #     mode='bilinear',
        #     align_corners=False)

        return batched_clip

    def extract_features_convnext(self, x):
        out = {}
        x = self.clip_model.visual.trunk.stem(x)
        out['stem'] = x.contiguous() # os4
        for i in range(4):
            x = self.clip_model.visual.trunk.stages[i](x)
            out[f'res{i+2}'] = x.contiguous() # res 2 (os4), 3 (os8), 4 (os16), 5 (os32)
        
        x = self.clip_model.visual.trunk.norm_pre(x)
        out['clip_vis_dense'] = x.contiguous()
        return out
    
    def extract_features_resnet(self, x):
        out = {}
        x = self.clip_model.visual.act1(self.clip_model.visual.bn1(self.clip_model.visual.conv1(x)))
        x = self.clip_model.visual.act2(self.clip_model.visual.bn2(self.clip_model.visual.conv2(x)))
        x = self.clip_model.visual.act3(self.clip_model.visual.bn3(self.clip_model.visual.conv3(x)))
        out['stem'] = x.contiguous() # os2
        x = self.clip_model.visual.avgpool(x)
        x = self.clip_model.visual.layer1(x)
        out['res2'] = x.contiguous() # os4
        x = self.clip_model.visual.layer2(x)
        out['res3'] = x.contiguous() # os8
        x = self.clip_model.visual.layer3(x)
        out['res4'] = x.contiguous() # os16
        x = self.clip_model.visual.layer4(x)
        out['res5'] = x.contiguous() # os32
        out['clip_vis_dense'] = x
        return out

    def extract_features(self, x):
        return {
            'convnext': self.extract_features_convnext,
            'resnet': self.extract_features_resnet,
        }[self.model_type](x)

    def forward(self, img_inputs):
        if isinstance(img_inputs, (tuple, list)):
            img_inputs, class_ids = img_inputs[:2]
            class_ids = class_ids.tolist()
        clip_inputs = self.clip_preprocess(img_inputs)
        # extract clip features
        if not self.finetune:
            with torch.no_grad():
                clip_features = self.extract_features(clip_inputs)
                # cls_token = self.clip_model.encode_image(clip_inputs)
        else:
            clip_features = self.extract_features(clip_inputs)
            # with torch.no_grad():
            #     cls_token = self.clip_model.encode_image(clip_inputs)            

        # calculate cross-attn map
        if self.class_embed_select:
            cls_token = self.class_embeddings[class_ids] # B x dim

        outputs = []
        for idx in range(len(self.clip_scales)):
            # fused_feature = inputs[idx] + F.interpolate(
            #     self.feature_fusion_layer[idx](clip_features[self.clip_scales[idx]].contiguous()),
            #     size=inputs[idx].shape[-2:],
            #     mode="bilinear",
            #     align_corners=False,
            # )
            fused_feature = clip_features[self.clip_scales[idx]].contiguous()
            outputs.append(fused_feature)
        if self.class_embed_select:
            return dict(visual=tuple(outputs), cls=cls_token)
        else:
            return tuple(outputs)
