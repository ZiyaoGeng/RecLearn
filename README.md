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

[简体中文](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/blob/reclearn/README_CN.md)｜[English](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/reclearn)

RecLearn (Recommender Learning)  which summarizes the contents of the [master](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master) branch in  `Recommender System with TF2.0 `  is a recommended learning framework based on Python and TensorFlow2.x for students and beginners. The implemented recommendation algorithms are classified according to two application stages in the industry:

- matching recommendation stage (Top-k Recmmendation)
- ranking  recommendeation stage (CTR predict model)

## Installation

RecLearn is on PyPI, so you can use `pip` to install it.

```
pip install reclearn
```

dependent environment：

- python3.7+
- Tensorflow2.6+
- sklearn

## Quick Start

In [example](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/reclearn/example), we have given a demo of each of the recommended models.

**Firstly，building dataset.**



**Then, constructing model.**



**Finally, Compile, Fit and Predict**



## Model List

### 1. Matching Stage

|                         Paper\|Model                         |  Published  |    Author     |
| :----------------------------------------------------------: | :---------: | :-----------: |
| BPR: Bayesian Personalized Ranking from Implicit Feedback\|**MF-BPR** |  UAI, 2009  | Steﬀen Rendle |
|    Neural network-based Collaborative Filtering\|**NCF**     |  WWW, 2017  |  Xiangnan He  |
|     Self-Attentive Sequential Recommendation｜**SASRec**     | ICDM, 2018  |     UCSD      |
| Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding｜**Caser** | WSDM, 2018  |  Jiaxi Tang   |
| Next Item Recommendation with Self-Attentive Metric Learning\|**AttRec** | AAAAI, 2019 |  Shuai Zhang  |

### 2. Ranking Stage

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

## Discussion

1. If you have any suggestions or questions about the project, you can leave a comment on `Issue` or email `zggzy1996@163.com`.
2. wechat：

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="20%"/></div>

