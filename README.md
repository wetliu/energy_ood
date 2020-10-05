# Energy-based Out-of-distribution Detection (Energy OOD)

This repository is the official implementation of [Energy-based Out-of-distribution Detection]()

![image](https://github.com/wetliu/energy_ood/blob/master/demo_figs/energy-ood.png)

## Pretrained Models

Please download the [pretrained models](https://github.com/hendrycks/outlier-exposure/tree/master/CIFAR/snapshots/baseline) and put them in folder

```
./CIFAR/snapshots/baseline/
```

## Testing and Fine-tuning

run energy score testing for cifar10 WRN
```test
bash run.sh energy 0
```

run energy score testing for cifar100 WRN
```test
bash run.sh energy 1
```

run energy score training and testing for cifar10 WRN
```train
bash run.sh energy_ft 0
```

run energy score training and testing for cifar100 WRN
```train
bash run.sh energy_ft 1
```

## Results

Our model achieves the following average performance on 6 OOD datasets:

### 1. MSP vs energy score with and without fine-tuned on [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)

| Model name         |     FPR95       |
| ------------------ |---------------- |
| Softmax score |     51.04%      |
| Energy score (ours)  |     33.01%      |
| Softmax score with fine-tune |     8.53%       |
| Energy score with fine-tune (ours) |     3.32%       |

### 2. CIFAR-10 (in-distribution) vs SVHN (out-of-distribution) Score Distributions

![image](https://github.com/wetliu/energy_ood/blob/master/demo_figs/cifar10_vs_svhn.png)

### 3. Performance among different baselines for [WideResNet](https://arxiv.org/abs/1605.07146)
CIFAR-10:
| Model name         |     FPR95       |
| ------------------ |---------------- |
| [Softmax score](https://arxiv.org/abs/1610.02136) |     51.04%      |
| Energy score (ours)  |     33.01%      |
| [ODIN](https://arxiv.org/abs/1706.02690)  |     35.71%       |
| [Mahalanobis](https://arxiv.org/abs/1807.03888) | 37.08% |
| [Outlier Exposure](https://arxiv.org/abs/1812.04606)|  8.53%    |
| Energy score with fine-tune (ours) |     3.32%       |

CIFAR-100:
| Model name         |     FPR95       |
| ------------------ |---------------- |
| [Softmax score](https://arxiv.org/abs/1610.02136) |     80.41%      |
| Energy score (ours)  |     73.60%      |
| [ODIN](https://arxiv.org/abs/1706.02690)  |     74.64%       |
| [Mahalanobis](https://arxiv.org/abs/1807.03888) | 54.64% |
| [Outlier Exposure](https://arxiv.org/abs/1812.04606)|   58.10%   |
| Energy score with fine-tune (ours) |     47.55%       |

## Outlier Datasets

These experiments make use of numerous outlier datasets. Links for less common datasets are as follows, [80 Million Tiny Images](http://horatio.cs.nyu.edu/mit/tiny/data/tiny_images.bin), [Icons-50](https://github.com/hendrycks/robustness),
[Textures](https://www.robots.ox.ac.uk/~vgg/data/dtd/), [Chars74K](http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/EnglishImg.tgz), and [Places365](http://places2.csail.mit.edu/download.html), [LSUN-C](https://www.dropbox.com/s/fhtsw1m3qxlwj6h/LSUN.tar.gz), [LSUN-R](https://www.dropbox.com/s/moqh2wh8696c3yl/LSUN_resize.tar.gz), [iSUN](https://www.dropbox.com/s/ssz7qxfqae0cca5/iSUN.tar.gz).

## Citation

     @article{liu2020energy,
          title={Energy-based Out-of-distribution Detection},
          author={Liu, Weitang and Wang, Xiaoyun and Owens, John and Li, Yixuan},
          journal={Advances in Neural Information Processing Systems},
          year={2020}
     } 
