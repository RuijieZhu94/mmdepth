from logging import raiseExceptions
import os.path as osp
import warnings
from collections import OrderedDict
from functools import reduce
from mmdepth.registry import DATASETS
from typing import List
from .basesegdataset import BaseSegDataset
from mmengine.logging import print_log

@DATASETS.register_module()
class IbimsDataset(BaseSegDataset):
    """DIODE dataset for depth estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── data
        │   ├── ibims/m1455541
        │   │   ├── ibims1_core_raw
        │   │   │   ├── calib
        │   │   │   ├── depth
        │   │   │   |   ...
        │   │   │   |   ...
        │   │   │   |   masked_depth
        │   │   │   |   rgb
        │   │   │   ├── imagelist.txt

    split file format:
    input_image: rgb/corridor_01.png
    gt_depth:    masked_depth/corridor_01.png

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
    METAINFO = dict()

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
        
    def load_data_list(self) -> List[dict]:
        """Load annotation from directory.
        Args:
            img_dir (str): Path to image directory
            ann_dir (str|None): Path to annotation directory.
            split (str|None): Split txt file. Split should be specified, only file in the splits will be loaded.
        Returns:
            list[dict]: All image info of dataset.
        """
        # img_dir = self.data_prefix.get('img_path', None)
        # ann_dir = self.data_prefix.get('depth_map_path', None)
        split = osp.join(self.data_root, self.split)
        self.invalid_depth_num = 0
        img_infos = []
        if split is not None:
            with open(split) as f:
                for line in f:
                    img_info = dict()
                    basename = line.strip().split(" ")[0]
                    # if ann_dir is not None: # benchmark test or unsupervised future
                    depth_map = osp.join(self.data_root, 'masked_depth', basename + ".png")
                    if depth_map == 'None':
                        self.invalid_depth_num += 1
                        continue
                    img_info['depth_map_path'] = depth_map
                    img_name = osp.join(self.data_root, 'rgb', basename + ".png")
                    img_info['img_path'] = img_name
                    img_info['seg_fields'] = []
                    img_info['category_id'] = -1
                    img_infos.append(img_info)
        else:
            print("Split should be specified, NotImplementedError")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        print_log(f'Loaded {len(img_infos)} images from Ibims dataset.'
                  'Totally {} invalid pairs are filtered'.format(self.invalid_depth_num),
                  logger='current')

        return img_infos


