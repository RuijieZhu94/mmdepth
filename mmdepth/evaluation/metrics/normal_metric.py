import os.path as osp
from collections import OrderedDict, defaultdict
from typing import Dict, List, Optional, Sequence

import cv2
import numpy as np
import torch
from mmengine.dist import is_main_process
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger, print_log
from mmengine.utils import mkdir_or_exist
from prettytable import PrettyTable
from torch import Tensor

from mmdepth.registry import METRICS


@METRICS.register_module()
class NormalMetric(BaseMetric):
    """Normal estimation evaluation metric.

    Args:
        normal_metrics (List[str], optional): List of metrics to compute. If
            not specified, defaults to all metrics in self.METRICS.
        min_normal_eval (float): Minimum normal value for evaluation.
            Defaults to 0.0.
        max_normal_eval (float): Maximum normal value for evaluation.
            Defaults to infinity.
        crop_type (str, optional): Specifies the type of cropping to be used
            during evaluation. This option can affect how the evaluation mask
            is generated. Currently, 'nyu_crop' is supported, but other
            types can be added in future. Defaults to None if no cropping
            should be applied.
        normal_scale_factor (float): Factor to scale the normal values.
            Defaults to 1.0.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        output_dir (str): The directory for output prediction. Defaults to
            None.
        format_only (bool): Only format result for results commit without
            perform evaluation. It is useful when you want to save the result
            to a specific format and submit it to the test server.
            Defaults to False.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonymous metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Defaults to None.
    """
    METRICS = ('a1', 'a2', 'a3', 'a4', 'a5', 'rmse_angular', 'mean', 'median')

    def __init__(self,
                 normal_metrics: Optional[List[str]] = None,
                 min_normal_eval: float = -1.0,
                 max_normal_eval: float = 1.0,
                 normal_clamp: bool = False,
                 gt_normal_clamp: bool = False,
                 focal_length_rescale: bool = False,
                 base_focal_length: float = 519.163756,
                 crop_type: Optional[str] = None,
                 normal_scale_factor: float = 1.0,
                 collect_device: str = 'cpu',
                 output_dir: Optional[str] = None,
                 format_only: bool = False,
                 prefix: Optional[str] = None,
                 **kwargs) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)

        if normal_metrics is None:
            self.metrics = self.METRICS
        elif isinstance(normal_metrics, [tuple, list]):
            for metric in normal_metrics:
                assert metric in self.METRICS, f'the metric {metric} is not ' \
                    f'supported. Please use metrics in {self.METRICS}'
            self.metrics = normal_metrics

        # Validate crop_type, if provided
        assert crop_type in [
            None, 'nyu_crop', 'eigen_crop', 'garg_crop'
        ], (f'Invalid value for crop_type: {crop_type}. Supported values are '
            'None or \'nyu_crop\'.')
        self.crop_type = crop_type
        self.min_normal_eval = min_normal_eval
        self.max_normal_eval = max_normal_eval
        self.normal_clamp = normal_clamp
        self.gt_normal_clamp = gt_normal_clamp
        self.output_dir = output_dir
        if self.output_dir and is_main_process():
            mkdir_or_exist(self.output_dir)
        self.format_only = format_only
        self.normal_scale_factor = normal_scale_factor
        self.base_focal_length = base_focal_length
        self.focal_length_rescale = focal_length_rescale

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data and data_samples.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            pred_label = data_sample['pred_depth_map']['data'].squeeze()
            if self.focal_length_rescale:
                pred_label = pred_label * data_sample['focal_length'] / self.base_focal_length
            if self.normal_clamp:
                pred_label[pred_label > self.max_normal_eval] = self.max_normal_eval
                pred_label[pred_label < self.min_normal_eval] = self.min_normal_eval

            # format_only always for test dataset without ground truth
            if not self.format_only:
                gt_normal = data_sample['gt_normal_map']['data'].squeeze().to(
                    pred_label)
                gt_normal_mask = data_sample['gt_normal_mask']['data'].squeeze().to(
                    pred_label)
                if self.gt_normal_clamp:
                    gt_normal[gt_normal > self.max_normal_eval] = self.max_normal_eval
                    gt_normal[gt_normal < self.min_normal_eval] = self.min_normal_eval

                eval_mask = self._get_eval_mask(gt_normal, gt_normal_mask) 
                self.results.append(
                    (gt_normal[:, eval_mask], pred_label[:, eval_mask]))
            # format_result
            if self.output_dir is not None:
                basename = data_sample['img_path'].replace('/', '_').replace('.jpg', '.png')
                png_filename = osp.abspath(
                    osp.join(self.output_dir, f'{basename}'))
                output_mask = pred_label.cpu().numpy(
                ) * self.normal_scale_factor

                cv2.imwrite(png_filename, output_mask.astype(np.uint16),
                            [cv2.IMWRITE_PNG_COMPRESSION, 0])

    def _get_eval_mask(self, gt_normal: Tensor, valid_mask: Optional[Tensor] = None):
        """Generates an evaluation mask based on ground truth normal and
        cropping.

        Args:
            gt_normal (Tensor): Ground truth normal map.

        Returns:
            Tensor: Boolean mask where evaluation should be performed.
        """
        if valid_mask is None:
            valid_mask = torch.zeros_like(gt_normal[0])

        if self.crop_type == 'nyu_crop':
            # this implementation is adapted from
            # https://github.com/zhyever/Monocular-normal-Estimation-Toolbox/blob/main/normal/datasets/nyu.py  # noqa
            crop_mask = torch.zeros_like(valid_mask)
            crop_mask[45:471, 41:601] = 1
        elif self.crop_type == 'eigen_crop':
            gt_height, gt_width = gt_normal.shape[-2:]
            assert gt_height==352 and gt_width==1216, 'do kb crop first'
            # eigen crop
            crop_mask = torch.zeros_like(valid_mask)
            crop_mask[int(0.3324324 * 352):int(0.91351351 * 352),
                      int(0.0359477 * 1216):int(0.96405229 * 1216)] = 1
        elif self.crop_type == 'garg_crop':
            gt_height, gt_width = gt_normal.shape[-2:]
            assert gt_height==352 and gt_width==1216, 'do kb crop first'
            # garg crop
            crop_mask = torch.zeros_like(valid_mask)
            crop_mask[int(0.40810811 * 352):int(0.99189189 * 352),
                      int(0.03594771 * 1216):int(0.96405229 * 1216)] = 1
        else:
            crop_mask = torch.ones_like(valid_mask)

        eval_mask = torch.logical_and(valid_mask, crop_mask)
        return eval_mask

    @staticmethod
    def _calc_all_metrics(gt_normal, pred_normal):
        """Computes final evaluation metrics based on accumulated results."""
        assert gt_normal.shape == pred_normal.shape

        prediction_error = torch.cosine_similarity(gt_normal, pred_normal, dim=0)
        prediction_error = torch.clamp(prediction_error, min=-1.0, max=1.0)
        err = torch.acos(prediction_error) * 180.0 / torch.pi

        a1 = torch.sum(err < 5).float() / len(err)
        a2 = torch.sum(err < 7.5).float() / len(err)
        a3 = torch.sum(err < 11.5).float() / len(err)
        a4 = torch.sum(err < 22.5).float() / len(err)
        a5 = torch.sum(err < 30.0).float() / len(err)

        rmse_angular = torch.sqrt(torch.mean(torch.pow(err, 2) + 1e-6))
        mean = torch.mean(err)
        median = torch.median(err)

        return {
            'a1': a1.item(),
            'a2': a2.item(),
            'a3': a3.item(),
            'a4': a4.item(),
            'a5': a5.item(),
            'rmse_angular': rmse_angular.item(),
            'mean': mean.item(),
            'median': median.item()
        }

    def compute_metrics(self, results: list) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (list): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
                the metrics, and the values are corresponding results. The keys
                are identical with self.metrics.
        """
        logger: MMLogger = MMLogger.get_current_instance()
        if self.format_only:
            logger.info(f'results are saved to {osp.dirname(self.output_dir)}')
            return OrderedDict()

        metrics = defaultdict(list)
        for gt_normal, pred_normal in results:
            for key, value in self._calc_all_metrics(gt_normal,
                                                     pred_normal).items():
                metrics[key].append(value)
        metrics = {k: sum(metrics[k]) / len(metrics[k]) for k in self.metrics}

        table_data = PrettyTable()
        for key, val in metrics.items():
            table_data.add_column(key, [round(val, 5)])

        print_log('results:', logger)
        print_log('\n' + table_data.get_string(), logger=logger)

        return metrics
