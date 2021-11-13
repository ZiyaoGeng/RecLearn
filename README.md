<div>
  <img src='https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/logo.jpg' width='36%'/>
</div>

## RecLearn

<p align="left">
  <img src='https://img.shields.io/badge/python-3.7+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-2.6+-blue'>
  <img src='https://img.shields.io/badge/License-MIT-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
  <img src='https://img.shields.io/badge/sklearn-0.23.2-brightgreen'>
</p>  

RecLearn对`Recommender System with TF2.0`中`master`分支的内容进行了归纳、整理，是一个基于Python和Tensorflow2.x开发的推荐学习框架，适合学生、初学者研究使用。实现的推荐算法按照以下两个阶段进行分类：

- matching recommendation stage
- ranking  recommendeation stage

&nbsp;

## 安装

RecLearn上传在pypi上，可以使用`pip`进行安装：

```
python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps reclearn

pip install reclearn
```



## 快速开始

在`example`中，给出了每一个推荐模型的demo。

首先，构建数据集。

然后，建立模型。

最后，编译、学习以及预测。

&nbsp;



## 复现论文列表

### 1. 召回模型（Top-K推荐）

|                         Paper\|Model                         |  Published  |    Author     |
| :----------------------------------------------------------: | :---------: | :-----------: |
| BPR: Bayesian Personalized Ranking from Implicit Feedback\|**MF-BPR** |  UAI, 2009  | Steﬀen Rendle |
|    Neural network-based Collaborative Filtering\|**NCF**     |  WWW, 2017  |  Xiangnan He  |
|     Self-Attentive Sequential Recommendation｜**SASRec**     | ICDM, 2018  |     UCSD      |
| Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding｜**Caser** | WSDM, 2018  |  Jiaxi Tang   |
| Next Item Recommendation with Self-Attentive Metric Learning\|**AttRec** | AAAAI, 2019 |  Shuai Zhang  |

&nbsp;

### 2. 排序模型（CTR预估）

|                         Paper｜Model                         |  Published   |                            Author                            |
| :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: |
|                Factorization Machines\|**FM**                |  ICDM, 2010  |                        Steffen Rendle                        |
| Field-aware Factorization Machines for CTR Prediction｜**FFM** | RecSys, 2016 |                       Criteo Research                        |
|    Wide & Deep Learning for Recommender Systems｜**WDL**     |  DLRS, 2016  |                         Google Inc.                          |
| Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features\|**Deep Crossing** |  KDD, 2016   |                      Microsoft Research                      |
| Product-based Neural Networks for User Response Prediction\|**PNN** |  ICDM, 2016  |                Shanghai Jiao Tong University                 |
|    Deep & Cross Network for Ad Click Predictions｜**DCN**    | ADKDD, 2017  |               Stanford University｜Google Inc.               |
| Neural Factorization Machines for Sparse Predictive Analytics\|**NFM** | SIGIR, 2017  |                         Xiangnan He                          |
| Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks\|**AFM** | IJCAI, 2017  |    Zhejiang University\|National University of Singapore     |
| DeepFM: A Factorization-Machine based Neural Network for CTR Prediction\|**DeepFM** | IJCAI, 2017  | Harbin Institute of Technology\|Noah’s Ark Research Lab, Huawei |
| xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems\|**xDeepFM** |  KDD, 2018   |        University of Science and Technology of China         |
| Deep Interest Network for Click-Through Rate Prediction\|**DIN** |  KDD, 2018   |                        Alibaba Group                         |

&nbsp;

&nbsp;
