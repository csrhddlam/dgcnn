# Point Cloud Classification with Weight Standardization
This project is our implementation of [Weight Standardization](https://github.com/joe-siyuan-qiao/WeightStandardization) for point cloud classification with Dynamic Graph CNN (DGCNN). The project is forked from [dgcnn](https://github.com/WangYueFt/dgcnn). Their original README.md is appended at the end.

Weight Standardization is a simple reparameterization method for convolutional layers.
It enables micro-batch training with Group Normalization (GN) to match the performances of Batch Normalization (BN) trained with large-batch sizes.
Please see our [arXiv](https://arxiv.org/abs/1803.?????) report for the details.
If you find this project helpful, please consider citing our paper.
```
@article{weightstandardization,
  author    = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Yuille},
  title     = {Weight Standardization},
  journal   = {arXiv preprint arXiv:1803.?????},
  year      = {2019},
}
```
## Performances of Weight Standardization (WS) on ModelNet40

| Method  | Mean Class Accuracy | Overall Accuracy |
|---------|:-------------------:|:----------------:|
| GN      | 87.0                | 89.7             |
| GN + WS | 88.8                | 91.2             |
| BN      | 89.3                | 91.7             |
| BN + WS | 89.6                | 92.0             |

## Training
We provide the following training scripts to get the reported results. We use tensorflow 1.12.0 with python3.
``` bash
# GN
python train.py --no-ws
# GN + WS
python train.py
# BN
python train.py --norm bn --no-ws
# BN + WS
python train.py --norm bn
```

## License
MIT License

## Acknowledgement
Our group normalization code is borrowed from [GroupNorm-reproduce](https://github.com/ppwwyyxx/GroupNorm-reproduce).

# Original README for Dynamic Graph CNN for Learning on Point Clouds
We propose a new neural network module dubbed EdgeConv suitable for CNN-based high-level tasks on point clouds including classification and segmentation. EdgeConv is differentiable and can be plugged into existing architectures.

[[Project]](https://liuziwei7.github.io/projects/DGCNN) [[Paper]](https://arxiv.org/abs/1801.07829)

## Overview
`DGCNN` is the author's re-implementation of Dynamic Graph CNN, which achieves state-of-the-art performance on point-cloud-related high-level tasks including category classification, semantic segmentation and part segmentation.

<img src='./misc/demo_teaser.png' width=800>

Further information please contact [Yue Wang](https://www.csail.mit.edu/person/yue-wang) and [Yongbin Sun](https://autoid.mit.edu/people-2).

## Requirements
* [TensorFlow](https://www.tensorflow.org/)

## Point Cloud Classification
* Run the training script:
``` bash
python train.py
```
* Run the evaluation script after training finished:
``` bash
python evalutate.py

```

## Citation
Please cite this paper if you want to use it in your work,

        @article{dgcnn,
          title={Dynamic Graph CNN for Learning on Point Clouds},
          author={Yue Wang, Yongbin Sun, Ziwei Liu, Sanjay E. Sarma, Michael M. Bronstein, Justin M. Solomon},
          journal={arXiv preprint arXiv:1801.07829},
          year={2018}
        }

## License
MIT License

## Acknowledgement
This code is heavily borrowed from [PointNet](https://github.com/charlesq34/pointnet).
