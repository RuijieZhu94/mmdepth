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
# from .plane_param_layers import local_planar_guidance, AO, reduction_1x1, parameterized_disparity, pqrs2depth, depth2pqrs, custom_pqrs2depth
from .nd_utils import DN_to_depth, DN_to_distance
import math
import numpy as np
import cv2


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
class Plane2DepthDecodeHead(MMDET_Mask2FormerHead):
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
                 num_queries :int = 100,
                 **kwargs) -> None:
        super().__init__(num_queries=num_queries, **kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes
        self.ignore_index = ignore_index
        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, 1)
        
        self.coef_embed = nn.Sequential(
        nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
        nn.Linear(feat_channels, 3))

        # custom
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.binsformer = binsformer
        self.loss_decode = SigLoss(valid_mask=True, max_depth=max_depth, loss_weight=10)

        self.norm = 'softmax'
        # self.seed_to_coe = reduction_1x1(feat_channels, feat_channels // 2, self.max_depth)
        self.dn_to_distance = DN_to_distance(4, 480, 640)  #bs, inputH, input W

    def plane2depth(self, predictions_logits, predictions_normal, predictions_dist):
        pred_depths = []
        output_normal= []
        output_distance = []
        
        # NYU
        K = np.array([[518.8579 / 4.0, 0, 325.5824 / 4.0, 0],
                [0, 518.8579 / 4.0, 253.7362 / 4.0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1]], dtype=np.float32)
        
        # K = np.array([[50, 0, 300 , 0],
        # [0, 50, 100, 0],
        # [0, 0, 1, 0],
        # [0, 0, 0, 1]], dtype=np.float32)
        
        # # KITTI
        # K = np.array([[716.88 / 4.0, 0, 596.5593 / 4.0, 0],
        #         [0, 716.88 / 4.0, 149.854 / 4.0, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]], dtype=np.float32)
        # # K[0][2] -= (370 - 352) / 4.0
        # # K[1][2] -= (1224 - 1216) / 4.0
        
        inv_K = np.linalg.pinv(K)
        K = torch.from_numpy(K)
        inv_K = torch.from_numpy(inv_K)
        bs, _ , _ ,_ = predictions_logits[0].shape
        inv_K = inv_K.repeat(bs,1,1).cuda()

        
        for pred_logit, pred_normal, pred_dist in \
            zip(predictions_logits, predictions_normal, predictions_dist):
            
            if self.binsformer is False:
                pred_depth = F.relu(self.pred_depth(pred_logit)) + self.min_depth
            else:
                pred_logit = pred_logit.softmax(dim=1)
                
                bins = pred_dist.squeeze(dim=2)
                
                if self.norm == 'linear':
                    bins = torch.relu(bins)
                    eps = 0.1
                    bins = bins + eps
                elif self.norm == 'softmax':
                    bins = torch.softmax(bins, dim=1)
                else:
                    bins = torch.sigmoid(bins)
                bins = bins / bins.sum(dim=1, keepdim=True)

                bin_widths = bins  # .shape = N, dim_out
                bin_widths = nn.functional.pad(bin_widths, (1, 0), mode='constant', value=self.min_depth)
                bin_edges = torch.cumsum(bin_widths, dim=1)
                centers = 0.5 * (bin_edges[:, :-1] + bin_edges[:, 1:])
                n, dout = centers.size()
                centers = centers.contiguous().view(n, dout, 1)
                
                pred_params_4 = torch.cat([pred_normal, centers], dim=2)
                allpixel_nd = torch.einsum('bqhw,bqp->bphw', pred_logit, pred_params_4)
                allpixel_n = allpixel_nd[:,:3]
                allpixel_norm = F.normalize(allpixel_n, dim=1, p=2)
                allpixel_dist = allpixel_nd[:,3:] * self.max_depth

                b, c, h, w =  allpixel_n.shape
                
                device = allpixel_n.device  
                dn_to_depth = DN_to_depth(b, h, w).to(device)

                pred_depth = dn_to_depth(allpixel_norm, allpixel_dist, inv_K).clamp(0, self.max_depth)

            pred_depths.append(pred_depth)
            output_normal.append(allpixel_norm)
            output_distance.append(allpixel_dist)


        return pred_depths, output_normal, output_distance

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
        normal_gt = []
        inv_K = []
        inv_K_p = []
        for sample in batch_data_samples:
            depth_gt.append(sample.gt_depth_map.data)
            normal_gt.append(sample.gt_normal_map.data)
            # inv_K.append(sample.inv_K)
            # inv_K_p.append(sample.inv_K_p)
            # metainfo = sample.metainfo
            # metainfo['batch_input_shape'] = metainfo['img_shape']
            # sample.set_metainfo(metainfo)
        depth_gt = torch.stack(depth_gt, dim=0)
        normal_gt = torch.stack(normal_gt, dim=0)
        # inv_K = torch.stack(inv_K, dim=0).cuda()
        # inv_K_p = torch.stack(inv_K_p, dim=0).cuda()
        
        losses = dict()
        # forward
        all_mask_preds, all_coed_preds, all_dist_preds = self(x, batch_data_samples)
        pred_depths, output_normal, output_distance = self.plane2depth(all_mask_preds, all_coed_preds, all_dist_preds)

    

        # loss
        # losses = self.loss_by_feat(all_cls_scores, all_mask_preds,
        #                            batch_gt_instances, batch_img_metas)
        

        # caculate normal and distance loss
        # NYU
        K_p = np.array([[518.8579, 0, 325.5824, 0],
            [0, 518.8579, 253.7362, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]], dtype=np.float32)
        
        # # KITTI
        # K_p = np.array([[716.88, 0, 596.5593, 0],
        #         [0, 716.88, 149.854, 0],
        #         [0, 0, 1, 0],
        #         [0, 0, 0, 1]], dtype=np.float32)
        
        inv_K_p = np.linalg.pinv(K_p)
        inv_K_p = torch.from_numpy(inv_K_p)
        bs, _ , _ ,_ = pred_depths[-1].shape
        inv_K_p = inv_K_p.repeat(bs,1,1).cuda()
        mask = depth_gt > 0.1
        
        # normal_gt = torch.stack([normal_gt[:, 0], normal_gt[:, 2], normal_gt[:, 1]], 1)
        normal_gt_norm = F.normalize(normal_gt, dim=1, p=2)
        distance_gt = self.dn_to_distance(depth_gt, normal_gt_norm, inv_K_p)
        
        aux_weight_dict = {}
        if train_cfg["aux_loss"]:

            for index, weight in zip(train_cfg["aux_index"], train_cfg["aux_weight"]):
                depth = pred_depths[index]
                # normal_est_norm = output_normal[index]
                # distance_est = output_distance[index]
                
                depth = resize(
                    input=depth,
                    size=depth_gt.shape[2:],
                    mode='bilinear',
                    align_corners=self.align_corners,
                    warning=False)
                # normal_est_norm = resize(
                #     input=normal_est_norm,
                #     size=depth_gt.shape[2:],
                #     mode='bilinear',
                #     align_corners=self.align_corners,
                #     warning=False)
                # distance_est = resize(
                #     input=distance_est,
                #     size=depth_gt.shape[2:],
                #     mode='bilinear',
                #     align_corners=self.align_corners,
                #     warning=False)
                        
                if self.binsformer is False:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight
                    # normal_loss =  5 * ((1 - (normal_gt_norm * normal_est_norm).sum(1, keepdim=True)) * mask.float()).sum() / (mask.float() + 1e-7).sum()
                    # distance_loss = 0.25 * torch.abs(distance_gt[mask] - distance_est[mask]).mean()

                else:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight
                    # normal_loss =  weight * 5 * ((1 - (normal_gt_norm * normal_est_norm).sum(1, keepdim=True)) * mask.float()).sum() / (mask.float() + 1e-7).sum()
                    # distance_loss = weight * 0.25 * torch.abs(distance_gt[mask] - distance_est[mask]).mean()

                    # if self.classify:
                    #     cls = pred_classes[index]
                    #     loss_ce, _ = self.loss_class(cls, class_label)
                    #     aux_weight_dict.update({'aux_loss_ce' + f"_{index}": loss_ce})

                    # if self.with_chamfer_loss:
                    #     bin = pred_bins[index]
                    #     bins_loss = self.loss_chamfer(bin, depth_gt) * weight
                    #     aux_weight_dict.update({'aux_loss_chamfer' + f"_{index}": bins_loss})
                
                aux_weight_dict.update({'aux_loss_depth' + f"_{index}": depth_loss})
                # aux_weight_dict.update({'aux_loss_normal' + f"_{index}": normal_loss})
                # aux_weight_dict.update({'aux_loss_distance' + f"_{index}": distance_loss})
            losses.update(aux_weight_dict)
            
        
        normal_est_norm = resize(
            input=output_normal[-1],
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        distance_est = resize(
            input=output_distance[-1],
            size=depth_gt.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False)
        
        loss_normal = 5 * ((1 - (normal_gt_norm * normal_est_norm).sum(1, keepdim=True)) * mask.float()).sum() / (mask.float() + 1e-7).sum()
        loss_distance = torch.abs(distance_gt[mask] - distance_est[mask]).mean()
        

        # segment, planar_mask, dissimilarity_map = compute_seg(normal_est_norm, distance_est[:, 0])
        # loss_grad_normal, loss_grad_distance = get_smooth_ND(normal_est_norm, distance_est, planar_mask)
        # w_normal = 1
        # w_distance = 1
        
        # # losses["loss_planar_consistency"] =  w_normal * loss_grad_normal + w_distance * loss_grad_distance
        # losses["loss_grad_normal"] =  w_normal * loss_grad_normal
        # losses["loss_grad_distance"] =  w_distance * loss_grad_distance 
        losses["loss_normal"] = loss_normal
        losses["loss_distance"] = loss_distance
        
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
        

        # forward
        all_mask_preds, all_coed_preds, all_dist_preds = self(x, batch_data_samples)
        # inv_K = batch_img_metas[0]['inv_K'].cuda()
        # inv_K = inv_K.unsqueeze(0)
    
        
        pred_depths, output_normal, output_distance = self.plane2depth(all_mask_preds, all_coed_preds, all_dist_preds)
        
        
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

        
    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int]) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        dist_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # dist_mask_embed = self.dist_mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)
        # dist_mask_pred = torch.einsum('bqc,bchw->bqhw', dist_mask_embed, mask_feature)
        # dist_mask_pred = None
        normal_pred = self.coef_embed(decoder_out)
        attn_mask = F.interpolate(
            mask_pred,
            attn_mask_target_size,
            mode='bilinear',
            align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return  mask_pred, attn_mask, normal_pred, dist_pred

    def forward(self, x: List[Tensor],
                batch_data_samples: SampleList) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_size = x[0].shape[0]
        mask_features, multi_scale_memorys = self.pixel_decoder(x)
        # multi_scale_memorys (from low resolution to high resolution)
        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            mask = decoder_input.new_zeros(
                (batch_size, ) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.flatten(
                2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)
        # shape (num_queries, c) -> (batch_size, num_queries, c)
        query_feat = self.query_feat.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))
        query_embed = self.query_embed.weight.unsqueeze(0).repeat(
            (batch_size, 1, 1))


        mask_pred_list = []
        normal_pred_list = []
        dist_pred_list = []

        mask_pred, attn_mask, normal, distance = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])

        mask_pred_list.append(mask_pred)
        normal_pred_list.append(normal)
        dist_pred_list.append(distance)
        
        #visual_attn_mask = attn_mask[:,:-self.num_cls_query,:]  # do not mask cls query
        
        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            mask_sum = (attn_mask.sum(-1) != attn_mask.shape[-1]).unsqueeze(-1)
            attn_mask = attn_mask & mask_sum
            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None)
            mask_pred, attn_mask, normal, distance = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])


            mask_pred_list.append(mask_pred)
            # dist_mask_pred_list.append(dist_mask_pred)
            normal_pred_list.append(normal)
            dist_pred_list.append(distance)

            
        return mask_pred_list, normal_pred_list, dist_pred_list