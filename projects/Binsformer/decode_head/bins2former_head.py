# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModule

try:
    from mmdet.models.dense_heads import Mask2FormerHead as MMDET_Mask2FormerHead
except ModuleNotFoundError:
    MMDET_Mask2FormerHead = BaseModule
from mmdepth.models.utils import resize
from torch import Tensor
from mmdepth.registry import MODELS
from mmdepth.utils import ConfigType, SampleList
from mmdepth.structures.seg_data_sample import SegDataSample

# sigloss from Binsformer
class SigLoss(nn.Module):
    """SigLoss.

        We adopt the implementation in `Adabins <https://github.com/shariqfarooq123/AdaBins/blob/main/loss.py>`_.

    Args:
        valid_mask (bool): Whether filter invalid gt (gt > 0). Default: True.
        loss_weight (float): Weight of the loss. Default: 1.0.
        max_depth (int): When filtering invalid gt, set a max threshold. Default: None.
        warm_up (bool): A simple warm up stage to help convergence. Default: False.
        warm_iter (int): The number of warm up stage. Default: 100.
    """

    def __init__(self,
                 valid_mask=True,
                 loss_weight=1.0,
                 max_depth=None,
                 warm_up=False,
                 warm_iter=100):
        super(SigLoss, self).__init__()
        self.valid_mask = valid_mask
        self.loss_weight = loss_weight
        self.max_depth = max_depth

        self.eps = 0.001 # avoid grad explode

        # HACK: a hack implementation for warmup sigloss
        self.warm_up = warm_up
        self.warm_iter = warm_iter
        self.warm_up_counter = 0

    def sigloss(self, input, target):
        if self.valid_mask:
            valid_mask = target > 0.001
            if self.max_depth is not None:
                valid_mask = torch.logical_and(target > 0.001, target <= self.max_depth)
            input = input[valid_mask]
            target = target[valid_mask]
        
        if self.warm_up:
            if self.warm_up_counter < self.warm_iter:
                g = torch.log(input + self.eps) - torch.log(target + self.eps)
                g = 0.15 * torch.pow(torch.mean(g), 2)
                self.warm_up_counter += 1
                return torch.sqrt(g)

        g = torch.log(input + self.eps) - torch.log(target + self.eps)
        Dg = torch.var(g) + 0.15 * torch.pow(torch.mean(g), 2)
        return torch.sqrt(Dg)

    def forward(self, depth_pred, depth_gt):
        """Forward function."""
        
        loss_depth = self.loss_weight * self.sigloss(depth_pred, depth_gt)
        return loss_depth


@MODELS.register_module()
class Bins2FormerDecodeHead(MMDET_Mask2FormerHead):
    """Bins2Former head
    Args:
        binsformer (bool): Switch from the baseline method to Binsformer module. Default: False.
        align_corners (bool): Whether to apply align_corners mode to achieve upsample. Default: True.
        norm_cfg (dict|): Config of norm layers.
            Default: dict(type='BN', requires_grad=True).
        act_cfg (dict): Config of activation layers.
            Default: dict(type='LeakyReLU', inplace=True).
        dms_decoder (bool): Whether to apply a transfomer encoder before cross-attention with queries. Default: True.
        transformer_encoder (dict|None): General transfomer encoder config before cross-attention with queries. 
        positional_encoding (dict|None): Position encoding (p.e.) config.
        conv_dim (int): Temp feature dimension. Default: 256.
        index (List): Default indexes of input features from encoder/neck module. Default: [0,1,2,3,4]
        trans_index (List): Selected indexes of pixel-wise features to apply self-/cross- attention with transformer head.
        transformer_decoder(dict|None): Config of transformer decoder.
        with_chamfer_loss (bool): Whether to apply chamfer loss on bins distribution. Default: False
        loss_chamfer (dict|None): Config of the chamfer loss.
        classify (bool): Whether to apply scene understanding aux task. Default: True.
        class_num (int): class number for scene understanding aux task. Default: 25
        loss_class (dict): Config of scene classification loss. Default: dict(type='CrossEntropyLoss', loss_weight=1e-1).
        train_cfg (dict): Config of aux loss following most detr-like methods.
            Default: dict(aux_loss=True,),
    """
    def __init__(self,
                 num_classes: int = 150,
                 align_corners: bool = False,
                 ignore_index: int = 255,
                 min_depth: float = 1e-3,
                 max_depth: float = 10.,
                 binsformer: bool = True,
                 **kwargs) -> None:
        super().__init__(**kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, 1)

        # custom
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.binsformer = binsformer
        self.loss_decode = SigLoss(valid_mask=True, max_depth=max_depth, loss_weight=10)
        self.norm = 'softmax'

    def bin2depth(self, predictions_bins, predictions_logits):

        pred_bins = []
        pred_depths = []
        for item_bin, pred_logit in \
            zip(predictions_bins, predictions_logits):
            
            if self.binsformer is False:
                pred_depth = F.relu(self.pred_depth(pred_logit)) + self.min_depth
            else:
                bins = item_bin.squeeze(dim=2)
                
                if self.norm == 'linear':
                    bins = torch.relu(bins)
                    eps = 0.1
                    bins = bins + eps
                elif self.norm == 'softmax':
                    bins = torch.softmax(bins, dim=1)
                else:
                    bins = torch.sigmoid(bins)
                bins = bins / bins.sum(dim=1, keepdim=True)

                bin_widths = (self.max_depth - self.min_depth) * bins  # .shape = N, dim_out
                bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
                bin_edges = torch.cumsum(bin_widths, dim=1)
                centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
                n, dout = centers.size()
                centers = centers.contiguous().view(n, dout, 1, 1)

                pred_logit = pred_logit.softmax(dim=1)
                pred_depth = torch.sum(pred_logit * centers, dim=1, keepdim=True)
            
                pred_bins.append(bin_edges)

            pred_depths.append(pred_depth)

        return pred_depths, pred_bins

    def loss(self, x: Tuple[Tensor], batch_data_samples: SampleList,
             train_cfg: ConfigType) -> dict:
        """Perform forward propagation and loss calculation of the decoder head
        on the features of the upstream network.

        Args:
            x (tuple[Tensor]): Multi-level features from the upstream
                network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            train_cfg (ConfigType): Training config.

        Returns:
            dict[str, Tensor]: a dictionary of loss components.
        """
        # batch SegDataSample to InstanceDataSample
        # batch_gt_instances, batch_img_metas = self._seg_data_to_instance_data(
        #     batch_data_samples)
        if 'cls' in x:
            cls_token = x['cls']
            x = x['visual']
        else:
            cls_token = None        
        
        depth_gt = []
        for sample in batch_data_samples:
            depth_gt.append(sample.gt_depth_map.data)
            # metainfo = sample.metainfo
            # metainfo['batch_input_shape'] = metainfo['img_shape']
            # sample.set_metainfo(metainfo)
        depth_gt = torch.stack(depth_gt, dim=0)

        losses = dict()
        # forward
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        pred_depths, pred_bins = self.bin2depth(all_cls_scores, all_mask_preds)

        # loss
        # losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
        #                            batch_gt_instances, batch_img_metas)

        aux_weight_dict = {}

        if train_cfg["aux_loss"]:

            for index, weight in zip(train_cfg["aux_index"], train_cfg["aux_weight"]):
                depth = pred_depths[index]

                depth = resize(
                    input=depth,
                    size=depth_gt.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                
                if self.binsformer is False:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                else:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                    # if self.classify:
                    #     cls = pred_classes[index]
                    #     loss_ce, _ = self.loss_class(cls, class_label)
                    #     aux_weight_dict.update({'aux_loss_ce' + f"_{index}": loss_ce})

                    # if self.with_chamfer_loss:
                    #     bin = pred_bins[index]
                    #     bins_loss = self.loss_chamfer(bin, depth_gt) * weight
                    #     aux_weight_dict.update({'aux_loss_chamfer' + f"_{index}": bins_loss})
                
                aux_weight_dict.update({'aux_loss_depth' + f"_{index}": depth_loss})
            
            losses.update(aux_weight_dict)

        # main loss
        depth = pred_depths[-1]
        depth = resize(
            input=depth,
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)

        if self.binsformer is False:
            depth_loss = self.loss_decode(depth, depth_gt)
        else:
            depth_loss = self.loss_decode(depth, depth_gt)

            # if self.classify:
            #     cls = pred_classes[-1]
            #     loss_ce, acc = self.loss_class(cls, class_label) 
            #     losses["loss_ce"] = loss_ce
            #     for index, topk in enumerate(acc):
            #         losses["ce_acc_level_{}".format(index)] = topk

            # if self.with_chamfer_loss:
            #     bin = pred_bins[-1]
            #     bins_loss = self.loss_chamfer(bin, depth_gt)
            #     losses["loss_chamfer"] = bins_loss
        losses["loss_depth"] = depth_loss

        return losses

    def predict(self, x: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tuple[Tensor]:
        """Test without augmentaton.

        Args:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_img_metas (List[:obj:`SegDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_sem_seg`.
            test_cfg (ConfigType): Test config.

        Returns:
            Tensor: A tensor of segmentation mask.
        """
        if 'cls' in x:
            cls_token = x['cls']
            x = x['visual']
        else:
            cls_token = None

        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]
        
        all_cls_scores, all_mask_preds = self(x, batch_data_samples)

        pred_depths, pred_bins = self.bin2depth(all_cls_scores, all_mask_preds)
        depth = pred_depths[-1]
        
        if 'pad_shape' in batch_img_metas[0]:
            size = batch_img_metas[0]['pad_shape']
        else:
            size = batch_img_metas[0]['img_shape']

        depth = resize(
            input=depth,
            size=size,
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        return depth
        # mask_cls_results = all_cls_scores[-1]
        # mask_pred_results = all_mask_preds[-1]

        # # upsample masks
        # img_shape = batch_img_metas[0]['batch_input_shape']
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=img_shape,
        #     mode='bilinear',
        #     align_corners=False)

        # # semantic inference
        # cls_score = F.softmax(mask_cls_results, dim=-1)[..., :-1]
        # mask_pred = mask_pred_results.sigmoid()
        # seg_logits = torch.einsum('bqc,bqhw->bchw', cls_score, mask_pred)
        # return seg_logits