# from logging import raiseExceptions
import os.path as osp
from typing import List
from mmengine.logging import print_log
from mmdepth.registry import DATASETS
from .basesegdataset import BaseSegDataset


@DATASETS.register_module()
class SUNRGBDDataset(BaseSegDataset):
    """SUNRGBD dataset for depth estimation. An example of file structure
    is as followed.
    .. code-block:: none
        ├── root_path
        │   ├── SUNRGBD
        │   │   ├── kv1
        │   │   ├── kv2
        │   │   ├── ...
        |   ├── RUNRGBD_val_splits.txt

    split file format:
    input_image: SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.jpg 
    gt_depth:    SUNRGBD/kv2/kinect2data/000002_2014-05-26_14-23-37_260595134347_rgbf000103-resize/image/0000103.png 
    Args:
        pipeline (list[dict]): Processing pipeline
        ann_dir (str, optional): Path to annotation directory. Default: None
        split (str, optional): Split txt file. If split is specified, only
            file with suffix in the splits will be loaded. Otherwise, all
            images in img_dir/ann_dir will be loaded. Default: None
        data_root (str, optional): Data root for img_dir/ann_dir. Default:
            None.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.
    """
    METAINFO = dict(classes=('sunrgbd'))

    def __init__(self,
                 data_prefix=dict(
                     img_path='', depth_map_path=''),
                 img_suffix='.jpg',
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
            img_dir (str): Path to data directory
            img_suffix (str): Suffix of images.
            depth_map_suffix (str|None): Suffix of depth maps.
            split (str|None): Split txt file. If split is specified, only file
                with suffix in the splits will be loaded. Otherwise, all images
                in img_dir/ann_dir will be loaded. Default: None
        Returns:
            list[dict]: All image info of dataset.
        """
        self.invalid_depth_num = 0
        self.split = osp.join(self.data_root, self.split)
        img_infos = []
        if self.split is not None:
            with open(self.split) as f:
                for line in f:
                    img_info = dict()
                    depth_map = line.strip().split(" ")[1]
                    if depth_map == 'None':
                        self.invalid_depth_num += 1
                        continue
                    img_info['depth_map_path'] = osp.join(self.data_root, depth_map)
                    img_name = line.strip().split(" ")[0]
                    img_info['img_path'] = osp.join(self.data_root, img_name)
                    img_info['focal_length'] = float(line.strip().split(" ")[2])
                    img_info['seg_fields'] = []
                    img_info['category_id'] = -1
                    img_infos.append(img_info)
        else:
            print("Split is None, ERROR")
            raise NotImplementedError 

        # github issue:: make sure the same order
        img_infos = sorted(img_infos, key=lambda x: x['img_path'])
        print_log(f'Loaded {len(img_infos)} images. Totally {self.invalid_depth_num} invalid pairs are filtered')
        return img_infos