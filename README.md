<div>
  <img src='https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/logo.jpg' width='36%'/>
</div>

## RecLearn

<p align="left">
  <img src='https://img.shields.io/badge/python-3.7+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-2.7+-blue'>
  <img src='https://img.shields.io/badge/License-MIT-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
  <img src='https://img.shields.io/badge/sklearn-0.23.2-brightgreen'>
</p>  

[简体中文](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/blob/reclearn/README_CN.md) | [English](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/reclearn)

RecLearn (Recommender Learning)  which summarizes the contents of the [master](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master) branch in  `Recommender System with TF2.0 `  is a recommended learning framework based on Python and TensorFlow2.x for students and beginners. **Of course, if you are more comfortable with the master branch, you can clone the entire package, run some algorithms in example, and also update and modify the content of model and layer**. The implemented recommendation algorithms are classified according to two application stages in the industry:

- matching recommendation stage (Top-k Recmmendation)
- ranking  recommendeation stage (CTR predict model)



## Installation

### Package

RecLearn is on PyPI, so you can use `pip` to install it.

```
pip install reclearn
```

dependent environment：

- python3.7+
- Tensorflow2.7+（**It is very important**)
- sklearn



### Local

Clone Reclearn to local:

```shell
git clone -b reclearn git@github.com:ZiyaoGeng/Recommender-System-with-TF2.0.git
```



## Quick Start

In [example](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/reclearn/example), we have given a demo of each of the recommended models.

**Firstly，building dataset.**

**Then, constructing model.**

**Finally, Compile, Fit and Predict**



## Results

### Matching

<table style="text-align:center">
  <tr></tr>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="3">ml-1m</th>
    <th colspan="3">Beauty</th>
    <th colspan="3">STEAM</th>
  </tr>
  <tr>
    <th>HR@10</th><th>MRR@10</th><th>NDCG@10</th>
    <th>HR@10</th><th>MRR@10</th><th>NDCG@10</th>
    <th>HR@10</th><th>MRR@10</th><th>NDCG@10</th>
  </tr>
  <tr><td>BPR</td><td>0.5768</td><td>0.2392</td><td>0.3016</td><td>0.7728</td><td>0.4220</td><td>0.5054</td><td>0.6160</td><td>0.3427</td><td>0.4074</td></tr>
  <tr><td>NCF</td><td>0.5711</td><td>0.2112</td><td>0.2950</td><td>0.7768</td><td>0.4273</td><td>0.5103</td><td>0.6164</td><td>0.2948</td><td>0.3706</td></tr>
  <tr><td>SASRec</td><td>0.8103</td><td>0.4812</td><td>0.5605</td><td>0.8606</td><td>0.5669</td><td>0.6374</td><td>0.6705</td><td>0.3532</td><td>0.4286</td></tr>
</table>




### Ranking

<table style="text-align:center">
  <tr></tr>
  <tr>
    <th rowspan="2">Model</th>
    <th colspan="2">500w(Criteo)</th>
    <th colspan="2">Criteo</th>
  </tr>
  <tr>
    <th>Log Loss</th>
    <th>AUC</th>
    <th>Log Loss</th>
    <th>AUC</th>
  </tr>
  <tr><td>FM</td><td>0.4765</td><td>0.7783</td><td>0.4762</td><td>0.7875</td></tr>
  <tr><td>FFM</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>WDL</td><td>0.4684</td><td>0.7822</td><td>0.4692</td><td>0.7930</td></tr>
  <tr><td>Deep Crossing</td><td>0.4670</td><td>0.7826</td><td>0.4693</td><td>0.7935</td></tr>
  <tr><td>PNN</td><td>-</td><td>0.7847</td><td>-</td><td>-</td></tr>
  <tr><td>DCN</td><td>-</td><td>0.7823</td><td>0.4691</td><td>0.7929</td></tr>
  <tr><td>NFM</td><td>0.4773</td><td>0.7762</td><td>0.4723</td><td>0.7889</td></tr>
  <tr><td>AFM</td><td>0.4819</td><td>0.7808</td><td>0.4692</td><td>0.7871</td></tr>
  <tr><td>DeepFM</td><td>-</td><td>0.7828</td><td>0.4650</td><td>0.8007</td></tr>
  <tr><td>xDeepFM</td><td>0.4690</td><td>0.7839</td><td>0.4696</td><td>0.7919</td></tr>
</table>



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

