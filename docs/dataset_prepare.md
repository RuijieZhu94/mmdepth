## Prepare datasets

It is recommended to symlink the dataset root to `$mmdepth/data`.
If your folder structure is different, you may need to change the corresponding paths in config files.

```none
mmdepth
├── data
│   ├── ddad 
│   ├── diml 
│   ├── DIODE 
│   ├── hypersim 
│   ├── ibims
│   ├── kitti 
│   ├── nyu 
│   ├── sunrgbd
│   └── vkitti2.0 
```

### **KITTI**

Download the offical dataset from this [link](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction), including the raw data (about 200G) and fine-grained ground-truth depth maps. 

Then, unzip the files into data/kitti. Remember to organizing the directory structure following instructions (Only need a few cut operations). 

Finally, copy split files (whose names are started with *kitti*) in splits folder into data/kitti. Here, I utilize eigen splits following other supervised methods.

Some methods may use the camera intrinsic parameters (*i.e.,* BTS), you need to download the [benchmark_cam](https://drive.google.com/file/d/1ktSDTUx9dDViBKoAeqTERTay1813xfUK/view?usp=sharing) consisting of camera intrinsic parameters of the benchmark test set.

### **NYU-Depth V2**

Following previous work, I utilize about 50K image-depth pairs as our training set and standard 652 images as the validation set. You can download the subset with the help of codes provided in [BTS](https://github.com/cleinc/bts/tree/master/pytorch).

```shell
$ git clone https://github.com/cleinc/bts.git
$ cd bts
$ python utils/download_from_gdrive.py 1AysroWpfISmm-yRFGBgFTrLy6FjQwvwP sync.zip
$ unzip sync.zip
```

Then, you need to download the standard test set from this [link](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html). (**Note**: The downloaded file will be unzipped to folder test and train. You need to cut the files in the test folder out to data/nyu, organizing the directory structure following the file trees provided on the top of this page.)

Finally, copy nyu_train.txt and nyu_test.txt in the splits folder into the data/nyu.


### **SUNRGBD**

The dataset can be download from this [link](https://rgbd.cs.princeton.edu/). 

## **Ibim-1 benchmark**

The dataset can be download from this [link](https://www.asg.ed.tum.de/lmf/ibims1/). 

## **DIODE**

The dataset can be download from this [link](https://diode-dataset.org/). 

## **Hypersim**

The dataset can be download from this [link](https://github.com/apple/ml-hypersim). 

## **VKITTI2.0**

The dataset can be download from this [link](https://europe.naverlabs.com/research/computer-vision/proxy-virtual-worlds-vkitti-2/). 

## DIML

The dataset can be download from this [link](https://dimlrgbd.github.io/rawdata). 

### **Custom Dataset**

We also provide a simple custom dataset class for users in `depth/datasets/custom.py`. Organize your data folder as our illustration. Note that instead of utilizing a split file to divide the train/val set, we directly classify data into train/val folder. A simple config file can be like:

```
train=dict(
    type=dataset_type,
    pipeline=dict(...),
    data_root='data/custom_dataset',
    test_mode=False,
    min_depth=1e-3,
    max_depth=10,
    depth_scale=1)
```

As for the custom dataset, we do not implement the evaluation details. If you want to get a quantitive metric result, you need to implement the `pre_eval` and `evaluate` functions following the ones in KITTI or other datasets.