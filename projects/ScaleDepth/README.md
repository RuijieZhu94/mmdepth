<h3 align="center"><strong>ScaleDepth: Decomposing Metric Depth Estimation into Scale Prediction and Relative Depth Estimation</strong></h3>

  <p align="center">
    <a href="https://ruijiezhu94.github.io/ruijiezhu/">Ruijie Zhu</a>,
    <a href="https://chuxwa.github.io/">Chuxin Wang</a>,
    <a href="https://indu1ge.github.io/ziyangsong">Ziyang Song</a>,
    <a href="https://github.com/lliu00">Li Liu</a>,
    <a href="https://github.com/Hevans123">Jianfeng He</a>,<br>
    <a href="http://staff.ustc.edu.cn/~tzzhang/">Tianzhu Zhang</a>,
    <a href="https://dblp.org/pid/z/YongdongZhang.html">Yongdong Zhang</a>,
    <br>
    University of Science and Technology of China
    <br>
    <b>Arxiv 2024</b>

</p>

<div align="center">
 <a href='https://arxiv.org/abs/2407.08187'><img src='https://img.shields.io/badge/Paper-arXiv-red'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<!-- <a href='https://arxiv.org/abs/[]'><img src='https://img.shields.io/badge/arXiv-[]-b31b1b.svg'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; -->
 <a href='https://ruijiezhu94.github.io/ScaleDepth'><img src='https://img.shields.io/badge/Project-Page-Green'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
 <a href='https://github.com/RuijieZhu94/ScaleDepth/tree/main?tab=Apache-2.0-1-ov-file'><img src='https://img.shields.io/badge/License-Apache 2.0-blue'></a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<img src="https://visitor-badge.laobi.icu/badge?page_id=ruijiezhu94.scaledepth" alt="visitor badge"/>
</div>




<p align="center">
<img src="../../assets/scaledepth_teaser.jpg" width="97%"/>
</p>

> Within a unified framework, our method ScaleDepth achieves both accurate indoor and outdoor metric depth estimation without setting depth ranges or finetuning models. Left: the input RGB image and corresponding depth prediction. Right: the comparison of model parameters and performance. With overall fewer parameters, our model ScaleDepth-NK significantly outperforms the state-of-the-art methods under same experimental settings.

<p align="center">
<img src="../../assets/scaledepth_zeroshot.jpg" width="97%"/>
</p>

> Without any finetuning, our model can generalize to scenes with different scales and accurately estimate depth from indoors to outdoors.

<p align="center">
<img src="../../assets/scaledepth.jpg" width="97%"/>
</p>

> The overall architecture of the proposed ScaleDepth. We design bin queries to predict relative depth distribution and scale queries to predict scene scale. During training, we preset text prompts containing 28 scene categories as input to the frozen CLIP text encoder. We then calculate the similarity between the updated scale queries and text embedding, and utilize the scene category as its auxiliary supervision. During inference, only a single image is required to obtain the relative depth and scene scale, thereby synthesizing a metric depth map.



## Installation

Please refer to [get_started.md](../../docs/get_started.md#installation) for installation and [dataset_prepare.md](docs/dataset_prepare.md#prepare-datasets) for dataset preparation.


You may also need to install these packages:
```shell
pip install "mmdet>=3.0.0rc4"
pip install open_clip_torch
pip install future tensorboard
pip install -r requirements/albu.txt
```

## Training and Inference

We provide [train.md](docs/train.md) and [inference.md](docs/inference.md) for the instruction of training and inference. 


projects/ScaleDepth/pretrained_weights/nyu_kitti_class_embeddings_convnext_large_d_320.pth

## Bibtex

If you like our work and use the codebase or models for your research, please cite our work as follows.

```
@ARTICLE{zhu2024scale,
  title={ScaleDepth: Decomposing Metric Depth Estimation into Scale Prediction and Relative Depth Estimation}, 
  author={Zhu, Ruijie and Wang, Chuxin and Song, Ziyang and Liu, Li and Zhang, Tianzhu and Zhang, Yongdong},
  journal={arXiv preprint arXiv:2407.08187},
  year={2024}
}
```
## Acknowledgement
We thank Jianfeng He and Jiacheng Deng for their thoughtful and valuable suggestions.
We thank the authors of [Binsformer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox) and [Zoedepth](https://github.com/isl-org/ZoeDepth) for their code.
