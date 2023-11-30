# Maximal Correlation Regression (MCR) on MNIST 

This repository contains simple [Keras](https://keras.io/) and [Pytorch](https://pytorch.org/) implementations of [Maximal Correlation Regression](https://ieeexplore.ieee.org/abstract/document/8979352) (MCR), on the [MNIST](http://yann.lecun.com/exdb/mnist/) dataset. 


## Maximal Correlation Regression
[Maximal Correlation Regression [1]](https://ieeexplore.ieee.org/abstract/document/8979352) (MCR) is a regression analysis approach based on Hirschfeld-Gebelein-Rényi (HGR) maximal correlation. The basic idea is to represent the dependency between data variable $X$ and label $Y$ by their maximally correlated features $f^\ast(X)$ and $g^\ast(Y)$.



## Implementation
The implementation is based on the maximizing of H-score of features f and g[^1][^2]:

$$H(f, g) = \mathbb{E}[f^T(X)g(Y)] - \mathbb{E}[f^T(X)]\mathbb{E}[g(Y)] - \frac12 \mathrm{tr}\left(\mathbb{E}[f(X) f^T(X)]\mathbb{E}[g(Y)g^T(Y)]\right)$$

[^1]: The implementation in the repo uses an H-score on zero-mean features.
[^2]: A more advanced version, which learns structured features, can be found [here](https://github.com/XiangxiangXu/h-nest).

The network architecture is as follows [1, Figure 6]:

<img src="images/net.png" width="768">

The feature extractor $\mathtt{NN}_f$ is a simple CNN [1, Figure 2], and by default, the output feature dimension is $k = 128$:

<img src="images/cnn.png" width="512">

We can also compare the performance of MCR with the baseline method trained on **S**oftmax classifier with **L**og loss (SL). When trained on 1,000 samples and set feature dimension $k = 10$, the extracted features for two methods can be visualized by T-SNE [2] as (left: MCR, right: SL)

<p float="left">
<img src="images/mcr.png" width="400"> &nbsp;
<img src="images/sl.png" width="400">
</p>


### Dependencies
* [Keras](https://keras.io/)
or
* [Pytorch](https://pytorch.org/)

The two implementations are independent.

### Cite
If you use MCR in your work, please cite the original paper as:
```
@article{xu2020maximal,
  title={Maximal correlation regression},
  author={Xu, Xiangxiang and Huang, Shao-Lun},
  journal={IEEE Access},
  volume={8},
  pages={26591--26601},
  year={2020},
  publisher={IEEE}
}
```

### Related Algorithms
The method of optimizing H-score is also used for multi-modal feature extraction [3] and unsupervised feature extraction [4], with similar implementations.


### References 
[1] Xu, Xiangxiang, and Shao-Lun Huang. "Maximal correlation regression." IEEE Access 8 (2020): 26591-26601.

[2] Van der Maaten, Laurens, and Geoffrey Hinton. "Visualizing data using t-SNE." Journal of machine learning research 9.11 (2008).

[3] Wang, Lichen, Jiaxiang Wu, Shao-Lun Huang, Lizhong Zheng, Xiangxiang Xu, Lin Zhang, and Junzhou Huang. "An efficient approach to informative feature extraction from multimodal data." In Proceedings of the AAAI Conference on Artificial Intelligence, vol. 33, no. 01, pp. 5281-5288. 2019.

[4] Huang, Shao-Lun, Xiangxiang Xu, and Lizhong Zheng. "An information-theoretic approach to unsupervised feature selection for high-dimensional data." IEEE Journal on Selected Areas in Information Theory 1.1 (2020): 157-166.
