# Copyright (c) Ruijie Zhu. All rights reserved.
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
from ..loss import SigLoss, BinsChamferLoss, CELoss, SigPlusLoss
from mmdepth.models.losses.cross_entropy_loss import CrossEntropyLoss
from mmengine.runner import CheckpointLoader, load_checkpoint

class PixelWiseDotProduct(nn.Module):
    def __init__(self):
        super(PixelWiseDotProduct, self).__init__()

    def forward(self, x, K):
        n, c, h, w = x.size()
        _, cout, ck = K.size()
        assert c == ck, "Number of channels in x and Embedding dimension (at dim 2) of K matrix must match"
        y = torch.matmul(x.view(n, c, h * w).permute(0, 2, 1), K.permute(0, 2, 1))  # .shape = n, hw, cout
        return y.permute(0, 2, 1).view(n, cout, h, w)

class ScalePredictor(nn.Module):
    def __init__(self, in_channel):
        super(ScalePredictor, self).__init__()
        self.fc = nn.Linear(in_channel, 1)
        self.eps = 0.1

    def forward(self, x):
        x = self.fc(x)
        x = torch.relu(x) + self.eps
        return x


@MODELS.register_module()
class ScaleDepthDecodeHead(MMDET_Mask2FormerHead):
    """ScaleDepth head
    Args:
        num_class (int): Dimension of predicted depth. Default: 1.
        align_corners (bool): Whether to apply align_corners mode to achieve upsample. Default: False.
        min_depth (float): Minimum depth in depth loss. Default: 1e-3.
        max_depth (float): Maximum depth in depth loss. Default: 10.
        depthbins (bool): Whether to use depthbins to predict depth. Default: True.
        with_chamfer_loss (bool): Whether to apply chamfer loss on bins distribution. Default: False.
        with_classify (bool): Whether to apply classify loss on scale prediction. Default: False.
        with_scale (bool): Whether to adaptively predict scale (depth range). Default: False.
        class_embed_path (str): Path of class embeddings. Default: ''.
        num_cls_query (int): Number of scale queries. Default: 3.
        num_queries (int): Number of bin queries. Default: 100.
    """
    def __init__(self,
                 num_classes: int = 1,
                 align_corners: bool = False,
                 min_depth: float = 1e-3,
                 max_depth: float = 10.,
                 depthbins: bool = True,
                 with_chamfer_loss: bool = False,
                 with_classify: bool = False,
                 with_scale: bool = False,
                 class_embed_path: str = '',
                 num_cls_query: int = 3,
                 num_queries: int = 100,
                 **kwargs) -> None:
        super().__init__(num_queries=num_queries + num_cls_query, **kwargs)

        self.num_classes = num_classes
        self.align_corners = align_corners
        self.out_channels = num_classes

        feat_channels = kwargs['feat_channels']
        self.cls_embed = nn.Linear(feat_channels, 1)

        # custom
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.depthbins = depthbins
        self.norm = 'softmax'
        self.num_cls_query = num_cls_query
        self.with_chamfer_loss = with_chamfer_loss
        self.with_classify = with_classify
        self.with_scale = with_scale
        if self.with_chamfer_loss:
            self.loss_chamfer = BinsChamferLoss(loss_weight=1e-1)
        if self.with_classify:
            # class embeddings
            class_embeddings = CheckpointLoader.load_checkpoint(class_embed_path)
            self.register_buffer('class_embeddings', class_embeddings)
            self.loss_class = CrossEntropyLoss(loss_weight=1e-2)
            # self.classify = nn.Linear(feat_channels*num_cls_query, 27)
        if self.with_scale:
            self.scale_predictor = ScalePredictor(num_cls_query * feat_channels)
            self.loss_decode = SigPlusLoss(valid_mask=True, max_depth=max_depth, loss_weight=10)
        else:
            self.loss_decode = SigLoss(valid_mask=True, max_depth=max_depth, loss_weight=10)

    def bin2depth(self, predictions_bins, predictions_logits):
        pred_bins = []
        pred_depths = []
        for item_bin, pred_logit in \
            zip(predictions_bins, predictions_logits):
            
            if self.depthbins is False:
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

                if self.with_scale:
                    bin_widths = bins                    
                else:
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

    def calculate_depth_scale(self, cls_querys):
        depth_scales = []
        for cls_query in cls_querys:
            B = cls_query.shape[0]
            cls_query = cls_query.view(B, -1)
            depth_scale = self.scale_predictor(cls_query)
            depth_scales.append(depth_scale)

        return depth_scales

    def calculate_similarity(self, cls_querys):
        cls_probs = []
        for cls_query in cls_querys:
            B = cls_query.shape[0]
            cls_query = cls_query.view(B, -1) # B x C
            cls_prob = (100.0 * cls_query @ self.class_embeddings.T).softmax(dim=-1) # B x N
            # cls_prob = (100.0 * self.classify(cls_query)).softmax(dim=-1) # for ablation
            # depth_scale = torch.matmul(feature, kernel).view(B, -1)
            # depth_scale = torch.sigmoid(depth_scale)
            cls_probs.append(cls_prob)

        return cls_probs


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
        depth_gt = []
        class_label = []
        for sample in batch_data_samples:
            depth_gt.append(sample.gt_depth_map.data)
            class_label.append(sample.category_id)
            # metainfo = sample.metainfo
            # metainfo['batch_input_shape'] = metainfo['img_shape']
            # sample.set_metainfo(metainfo)
        depth_gt = torch.stack(depth_gt, dim=0)
        class_label = torch.LongTensor(class_label).cuda() # B

        losses = dict()

        if 'cls' in x:
            cls_token = x['cls']
            x = x['visual']

        # forward
        all_bins, all_mask_preds, cls_kernels = self(x, batch_data_samples)
        if self.with_scale:
            depth_scales = self.calculate_depth_scale(cls_kernels)
        if self.with_classify:
            pred_classes = self.calculate_similarity(cls_kernels)
        pred_depths, pred_bins = self.bin2depth(all_bins, all_mask_preds)

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
                
                if self.depthbins is False:
                    depth_loss = self.loss_decode(depth, depth_gt) * weight

                else:
                    if self.with_classify:
                        cls_score = pred_classes[index]
                        loss_ce = self.loss_class(cls_score, class_label)
                        aux_weight_dict.update({'aux_loss_ce' + f"_{index}": loss_ce})

                    if self.with_chamfer_loss:
                        bin = pred_bins[index]
                        bins_loss = self.loss_chamfer(bin, depth_gt) * weight
                        aux_weight_dict.update({'aux_loss_chamfer' + f"_{index}": bins_loss})

                    if self.with_scale:
                        depth_loss = self.loss_decode(depth, depth_gt, depth_scales[index]) * weight
                    else:
                        depth_loss = self.loss_decode(depth, depth_gt) * weight                
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

        if self.depthbins is False:
            depth_loss = self.loss_decode(depth, depth_gt)
        else:
            if self.with_classify:
                cls_score = pred_classes[-1]
                loss_ce = self.loss_class(cls_score, class_label) 
                losses["loss_ce"] = loss_ce
                # for index, topk in enumerate(acc):
                #     losses["ce_acc_level_{}".format(index)] = topk

            if self.with_chamfer_loss:
                bin = pred_bins[-1]
                bins_loss = self.loss_chamfer(bin, depth_gt)
                losses["loss_chamfer"] = bins_loss

            if self.with_scale:
                depth_loss = self.loss_decode(depth, depth_gt, depth_scales[-1])
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

        batch_data_samples = [
            SegDataSample(metainfo=metainfo) for metainfo in batch_img_metas
        ]
        
        all_cls_scores, all_mask_preds, cls_kernels = self(x, batch_data_samples)
        # if self.with_classify:
        #     pred_classes = self.calculate_similarity(cls_kernels)        
        pred_depths, pred_bins = self.bin2depth(all_cls_scores, all_mask_preds)
        depth = pred_depths[-1]
        if self.with_scale:
            depth_scales = self.calculate_depth_scale(cls_kernels)        
            depth *= depth_scales[-1].view(depth.shape[0], 1, 1, 1)    

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
        batch_img_metas = [
            data_sample.metainfo for data_sample in batch_data_samples
        ]
        batch_size = len(batch_img_metas)
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

        cls_pred_list = []
        mask_pred_list = []
        cls_kernel_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:])
        # cls_pred_list.append(cls_pred)
        # mask_pred_list.append(mask_pred)

        visual_attn_mask = attn_mask[:,:-self.num_cls_query,:]

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            visual_attn_mask[torch.where(
                visual_attn_mask.sum(-1) == visual_attn_mask.shape[-1])] = False
            attn_mask = F.pad(visual_attn_mask, (0, 0, 0, self.num_cls_query), "constant", False)
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
                key_padding_mask=None) # query_feat: B x N x C
            cls_kernel = query_feat[:,-self.num_cls_query:,:]
            visual_feat = query_feat[:,:-self.num_cls_query,:]
            cls_pred, mask_pred, visual_attn_mask = self._forward_head(
                visual_feat, mask_features, multi_scale_memorys[
                    (i + 1) % self.num_transformer_feat_level].shape[-2:])

            cls_kernel_list.append(cls_kernel)
            cls_pred_list.append(cls_pred)
            mask_pred_list.append(mask_pred)

        return cls_pred_list, mask_pred_list, cls_kernel_list