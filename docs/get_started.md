## Prerequisites
- Linux or macOS (Windows is in experimental support)
- Python 3.7+
- PyTorch 1.8+
- CUDA 10.2+

## Installation
I ran experiments with PyTorch 2.0.1, CUDA 11.7, Python 3.8, and Ubuntu 20.04. Other settings that satisfact the requirement would work.

### **If you have a similar environment**
You can simply follow our settings:

Use Anaconda to create a conda environment:

```shell
conda create -n mmdepth python=3.8
conda activate mmdepth
```

Install Pytorch following [official instructions](https://pytorch.org/get-started/locally/), e.g.,
```shell
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

Then, install [MMCV](https://github.com/open-mmlab/mmcv) using [MIM](https://github.com/open-mmlab/mim).

```shell
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
```

If you develop and run mmdepth directly, install it from source:

```shell
git clone -b main https://github.com/RuijieZhu94/mmdepth.git 
cd mmdepth
pip install -v -e .
# '-v' means verbose, or more output
# '-e' means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```

If training, you should install the tensorboard:
```shell
pip install future tensorboard
```


More information about installation can be found in docs of MMSegmentation (see [get_started.md](https://github.com/open-mmlab/mmsegmentation/blob/master/docs/en/get_started.md#installation)).

