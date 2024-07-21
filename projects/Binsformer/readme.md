# BinsFormer

[BinsFormer: Revisiting Adaptive Bins for Monocular Depth Estimation](https://arxiv.org/abs/2204.00987)

## Introduction

From https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox.

## Abstract

Monocular depth estimation is a fundamental task in computer vision and has drawn increasing attention. Recently, some methods reformulate it as a classification-regression task to boost the model performance, where continuous depth is estimated via a linear combination of predicted probability distributions and discrete bins. In this paper, we present a novel framework called BinsFormer, tailored for the classification-regression-based depth estimation. It mainly focuses on two crucial components in the specific task: (1) proper generation of adaptive bins and (2) sufficient interaction between probability distribution and bins predictions. To specify, we employ the Transformer decoder to generate bins, novelly viewing it as a direct set-to-set prediction problem. We further integrate a multi-scale decoder structure to achieve a comprehensive understanding of spatial geometry information and estimate depth maps in a coarse-to-fine manner. Moreover, an extra scene understanding query is proposed to improve the estimation accuracy, which turns out that models can implicitly learn useful information from an auxiliary environment classification task. Extensive experiments on the KITTI, NYU, and SUN RGB-D datasets demonstrate that BinsFormer surpasses state-of-the-art monocular depth estimation methods with prominent margins.


## Framework
<div align=center><img src="../../assets/binsformer.jpg"/></div>

## Training and Inference

We provide [train.md](docs/train.md) and [inference.md](docs/inference.md) for the instruction of training and inference. 

### Train
```shell
# BinsFormer - NYU
bash tools/dist_train.sh projects/Binsformer/configs/binsformer/binsformer_swinl_22k_NYU_416x544.py 4
```

### Test
```shell
# BinsFormer - NYU
python tools/test.py projects/Binsformer/configs/binsformer/binsformer_swinl_22k_NYU_416x544.py work_dirs/binsformer_swinl_22k_NYU_416x544/iter_40000.pth
```


## Citation

```bibtex
@article{li2022binsformer,
  title={BinsFormer: Revisiting Adaptive Bins for Monocular Depth Estimation},
  author={Li, Zhenyu and Wang, Xuyang and Liu, Xianming and Jiang, Junjun},
  journal={arXiv preprint arXiv:2204.00987},
  year={2022}
}
```

