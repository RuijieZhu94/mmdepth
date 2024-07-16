from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce

import mmcv
import numpy as np
from mmengine.logging import print_log
from prettytable import PrettyTable
from torch.utils.data import Dataset

# from mmdepth.evaluation.metrics import pre_eval_to_metrics, metrics, eval_metrics

from mmdepth.registry import DATASETS
from mmengine.dataset import Compose

from mmdepth.models.utils import resize

from PIL import Image
from typing import List
import torch
from .basesegdataset import BaseSegDataset

@DATASETS.register_module()
class KITTIBenchmarkDataset(BaseSegDataset):
    """KITTI dataset for depth estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── KITTI
        │   │   ├── kitti_eigen_train.txt
        │   │   ├── kitti_eigen_test.txt
        │   │   ├── input (RGB, img_dir)
        │   │   │   ├── date_1
        │   │   │   ├── date_2
        │   │   │   |   ...
        │   │   │   |   ...
        |   │   ├── gt_depth (ann_dir)
        │   │   │   ├── date_drive_number_sync
    split file format:
    input_image: 2011_09_26/2011_09_26_drive_0002_sync/image_02/data/0000000069.png 
    gt_depth:    2011_09_26_drive_0002_sync/proj_depth/groundtruth/image_02/0000000069.png 
    focal:       721.5377 (following the focal setting in BTS, but actually we do not use it)
    Args:
        pipeline (list[dict]): Processing pipeline
        img_dir (str): Path to image directory
        ann_dir (str, optional): Path to annotation directory. Default: None
        split (str, optional): Split txt file. Split should be specified, only file in the splits will be loaded.
        data_root (str, optional): Data root for img_dir/ann_dir. Default: None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
        depth_scale=256: Default KITTI pre-process. divide 256 to get gt measured in meters (m)
        garg_crop=True: Following Adabins, use grag crop to eval results.
        eigen_crop=False: Another cropping setting.
        min_depth=1e-3: Default min depth value.
        max_depth=80: Default max depth value.
    """
    METAINFO = dict(classes=('printer_room', 'bathroom', 'living_room', 'study',
                 'conference_room', 'study_room', 'kitchen', 'home_office',
                 'bedroom', 'dinette', 'playroom', 'indoor_balcony',
                 'laundry_room', 'basement', 'excercise_room', 'foyer',
                 'home_storage', 'cafe', 'furniture_store', 'office_kitchen',
                 'student_lounge', 'dining_room', 'reception_room',
                 'computer_lab', 'classroom', 'office', 'bookstore','outdoor_scene'))

    def __init__(self,
                 data_prefix=dict(
                     img_path='input', depth_map_path='gt_depth'),
                 img_suffix='.png',
                 depth_map_suffix='.png',
                 split=None,
                 data_root=None,
                 **kwargs) -> None:
        self.split = split
        super().__init__(
            data_prefix=data_prefix,
            img_suffix=img_suffix,
            seg_map_suffix=depth_map_suffix,
            data_root=data_root,
            **kwargs)

                #  pipeline,
                #  img_dir,
                #  ann_dir=None,
                #  split=None,
                #  data_root=None,
                #  test_mode=False,
                #  depth_scale=256,
                #  garg_crop=True,
                #  eigen_crop=False,
                #  min_depth=1e-3,
                #  max_depth=80):

        # self.pipeline = Compose(pipeline)
        # self.img_dir = data_prefix['img_path']
        # self.ann_dir = data_prefix['depth_map_path']

        # self.data_root = data_root

        # self.depth_scale = depth_scale
        # self.garg_crop = garg_crop
        # self.eigen_crop = eigen_crop
        # self.min_depth = min_depth # just for evaluate. (crop gt to certain range)
        # self.max_depth = max_depth # just for evaluate.

        # join paths if data_root is specified
        # if self.data_root is not None:
        #     if not (self.img_dir is None or osp.isabs(self.img_dir)):
        #         self.img_dir = osp.join(self.data_root, self.img_dir)
        #     if not (self.ann_dir is None or osp.isabs(self.ann_dir)):
        #         self.ann_dir = osp.join(self.data_root, self.ann_dir)
        #     if not (self.split is None or osp.isabs(self.split)):
        #         self.split = osp.join(self.data_root, self.split)

        # load annotations
        # self.img_infos = self.load_annotations(self.img_dir, self.ann_dir, self.split)
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """
        img_dir = self.data_prefix.get('img_path', None)
        ann_dir = self.data_prefix.get('depth_map_path', None)
        split = osp.join(self.data_root, self.split)
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    if ann_dir is not None: # benchmark test or unsupervised future
                        depth_map = line.strip().split(" ")[1]
                        if depth_map == 'None':
                            self.invalid_depth_num += 1
                            continue
                        img_info['depth_map_path'] = osp.join(ann_dir, depth_map) 
                    img_name = line.strip().split(" ")[0]
                    img_info['img_path'] = osp.join(img_dir, img_name)
                    img_info['seg_fields'] = []
                    img_info['category_id'] = self._metainfo['classes'].index('outdoor_scene')
                    img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        print_log(f'Loaded {len(img_infos)} images from KITTI benchmark.'
                  'Totally {} invalid pairs are filtered'.format(self.invalid_depth_num),
                  logger='current')

        return img_infos


    # def pre_pipeline(self, results):
    #     """Prepare results dict for pipeline."""
    #     results['depth_fields'] = []
    #     results['img_prefix'] = self.img_dir
    #     results['depth_prefix'] = self.ann_dir
    #     results['depth_scale'] = self.depth_scale

    #     results['cam_intrinsic_dict'] = {
    #         '2011_09_26' : [[7.215377e+02, 0.000000e+00, 6.095593e+02, 4.485728e+01], 
    #                         [0.000000e+00, 7.215377e+02, 1.728540e+02, 2.163791e-01],
    #                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.745884e-03]],
    #         '2011_09_28' : [[7.070493e+02, 0.000000e+00, 6.040814e+02, 4.575831e+01], 
    #                         [0.000000e+00, 7.070493e+02, 1.805066e+02, -3.454157e-01], 
    #                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 4.981016e-03]],
    #         '2011_09_29' : [[7.183351e+02, 0.000000e+00, 6.003891e+02, 4.450382e+01], 
    #                         [0.000000e+00, 7.183351e+02, 1.815122e+02, -5.951107e-01],
    #                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 2.616315e-03]],
    #         '2011_09_30' : [[7.070912e+02, 0.000000e+00, 6.018873e+02, 4.688783e+01], 
    #                         [0.000000e+00, 7.070912e+02, 1.831104e+02, 1.178601e-01], 
    #                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 6.203223e-03]],
    #         '2011_10_03' : [[7.188560e+02, 0.000000e+00, 6.071928e+02, 4.538225e+01], 
    #                         [0.000000e+00, 7.188560e+02, 1.852157e+02, -1.130887e-01], 
    #                         [0.000000e+00, 0.000000e+00, 1.000000e+00, 3.779761e-03]],
    #     }


    
    # def eval_kb_crop(self, depth_gt):
    #     """Following Adabins, Do kb crop for testing"""
    #     height = depth_gt.shape[0]
    #     width = depth_gt.shape[1]
    #     top_margin = int(height - 352)
    #     left_margin = int((width - 1216) / 2)
    #     depth_cropped = depth_gt[top_margin: top_margin + 352, left_margin: left_margin + 1216]
    #     depth_cropped = np.expand_dims(depth_cropped, axis=0)
    #     return depth_cropped

    # def eval_mask(self, depth_gt):
    #     """Following Adabins, Do grag_crop or eigen_crop for testing"""
    #     depth_gt = np.squeeze(depth_gt)
    #     valid_mask = np.logical_and(depth_gt > self.min_depth, depth_gt < self.max_depth)
    #     if self.garg_crop or self.eigen_crop:
    #         gt_height, gt_width = depth_gt.shape
    #         eval_mask = np.zeros(valid_mask.shape)

    #         if self.garg_crop:
    #             eval_mask[int(0.40810811 * gt_height):int(0.99189189 * gt_height),
    #                       int(0.03594771 * gt_width):int(0.96405229 * gt_width)] = 1

    #         elif self.eigen_crop:
    #             eval_mask[int(0.3324324 * gt_height):int(0.91351351 * gt_height),
    #                       int(0.0359477 * gt_width):int(0.96405229 * gt_width)] = 1
    #     valid_mask = np.logical_and(valid_mask, eval_mask)
    #     valid_mask = np.expand_dims(valid_mask, axis=0)
    #     return valid_mask

    # def pre_eval(self, preds, indices):
    #     """Collect eval result from each iteration.
    #     Args:
    #         preds (list[torch.Tensor] | torch.Tensor): the depth estimation.
    #         indices (list[int] | int): the prediction related ground truth
    #             indices.
    #     Returns:
    #         list[torch.Tensor]: (area_intersect, area_union, area_prediction,
    #             area_ground_truth).
    #     """
    #     # In order to compat with batch inference
    #     if not isinstance(indices, list):
    #         indices = [indices]
    #     if not isinstance(preds, list):
    #         preds = [preds]

    #     pre_eval_results = []
    #     pre_eval_preds = []

    #     for i, (pred, index) in enumerate(zip(preds, indices)):
    #         depth_map = osp.join(self.ann_dir,
    #                            self.img_infos[index]['ann']['depth_map'])

    #         depth_map_gt = np.asarray(Image.open(depth_map), dtype=np.float32) / self.depth_scale
    #         depth_map_gt = self.eval_kb_crop(depth_map_gt)
    #         valid_mask = self.eval_mask(depth_map_gt)
            
    #         eval = metrics(depth_map_gt[valid_mask], 
    #                        pred[valid_mask], 
    #                        min_depth=self.min_depth,
    #                        max_depth=self.max_depth)

    #         pre_eval_results.append(eval)

    #         # save prediction results
    #         pre_eval_preds.append(pred)

    #     return pre_eval_results, pre_eval_preds

    # def evaluate(self, results, metric='eigen', logger=None, **kwargs):
    #     """Evaluate the dataset.
    #     Args:
    #         results (list[tuple[torch.Tensor]] | list[str]): per image pre_eval
    #              results or predict depth map for computing evaluation
    #              metric.
    #         logger (logging.Logger | None | str): Logger used for printing
    #             related information during evaluation. Default: None.
    #     Returns:
    #         dict[str, float]: Default metrics.
    #     """
    #     metric = ["a1", "a2", "a3", "abs_rel", "rmse", "log_10", "rmse_log", "silog", "sq_rel"]
        
    #     eval_results = {}
    #     # test a list of files
    #     if mmcv.is_list_of(results, np.ndarray) or mmcv.is_list_of(
    #             results, str):
    #         gt_depth_maps = self.get_gt_depth_maps()
    #         ret_metrics = eval_metrics(
    #             gt_depth_maps,
    #             results)
    #     # test a list of pre_eval_results
    #     else:
    #         ret_metrics = pre_eval_to_metrics(results)
        
    #     ret_metric_names = []
    #     ret_metric_values = []
    #     for ret_metric, ret_metric_value in ret_metrics.items():
    #         ret_metric_names.append(ret_metric)
    #         ret_metric_values.append(ret_metric_value)

    #     num_table = len(ret_metrics) // 9
    #     for i in range(num_table):
    #         names = ret_metric_names[i*9: i*9 + 9]
    #         values = ret_metric_values[i*9: i*9 + 9]

    #         # summary table
    #         ret_metrics_summary = OrderedDict({
    #             ret_metric: np.round(np.nanmean(ret_metric_value), 4)
    #             for ret_metric, ret_metric_value in zip(names, values)
    #         })

    #         # for logger
    #         summary_table_data = PrettyTable()
    #         for key, val in ret_metrics_summary.items():
    #             summary_table_data.add_column(key, [val])

    #         print_log('Summary:', logger)
    #         print_log('\n' + summary_table_data.get_string(), logger=logger)

    #     # each metric dict
    #     for key, value in ret_metrics.items():
    #         eval_results[key] = value

    #     return eval_results
