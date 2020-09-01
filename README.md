<div>
  <img src='https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_0.png' width='40%'/>
</div>

## 前言

<p align="left">
  <img src='https://img.shields.io/badge/python-3.7-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
  <img src='https://img.shields.io/badge/sklearn-0.23.2-brightgreen'>
  <img src='https://img.shields.io/badge/Tensorflow-2.0-brightgreen'>
</p>  


开源项目`Recommender System with TF2.0`主要是对阅读过的部分推荐系统、CTR预估论文进行复现，包括**传统模型**（MF、FM、FFM等）、**神经网络模型**（WDL、DCN等）以及**序列模型**（DIN）。建立的**原因**有三个：

1. 理论和实践似乎有很大的间隔，学术界与工业界的差距更是如此；
2. 更好的理解论文的核心内容，增强自己的工程能力；
3. 很多论文给出的开源代码都是TF1.x，因此想要用更简单的TF2.0进行复现；

**项目特点：**

- 使用Tensorflow2.0进行复现；
- 每个模型都是相互独立的，不存在依赖关系；
- 模型基本按照论文进行构建，实验尽量使用论文给出的的公共数据集；
- 对于[实验数据集](#数据集介绍)有专门详细的介绍；
- 包含[模型结构文档](./~document/model_document.md)；
- 代码源文件参数、函数命名规范，并且带有标准的注释；



## 更新

2020.09.01：更新README；

2020.08.31：MF模型【采用显示反馈，在基础MF上加入bias】；

2020.08.26：FFM模型；【该模型**也仅仅是为了复现**，实际应用最好使用下述提到的C++包】

2020.08.25：FM模型；

2020.08.24：重写部分模型；

2020.08.21：xDeepFM模型；

2020.08.03：AFM模型；

2020.08.02：NFM模型；

2020.07.31：DeepFM模型；

2020.07.29：PNN代码文档更新；

2020.07.28：更改ReadMe介绍；

2020.07.27：Deep Crossing模型；

2020.07.20：PNN模型；

2020.07.14：DCN模型；

2020.07.10：Wide&Deep模型；

2020.05.26：DIN模型；

2020.03.27：NCF模型；



## 实验

1、通过git命令`git clone https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0.git`或者直接下载；

2、根据自己数据集的位置，合理更改所需模型文件内`train.py`的`file`路径；

3、设置`超参数`，直接运行即可；



## 数据集介绍

目前实验使用的数据集主要有三个：Movielens、Amazon、Criteo。

### Movielens

[MovieLens](https://grouplens.org/datasets/movielens/)是历史最悠久的推荐系统数据集，主要分为：ml-100k（1998年）、ml-1m（2003年）、ml-10m（2009年）、ml-20m（2015年）、ml-25m（2019年）。实验中主要使用ml-1m数据集。

已处理过的数据集：[ml-1m](https://github.com/hexiangnan/neural_collaborative_filtering)

ml-1m数据集的具体介绍与处理：[传送门](~document/Dataset%20Introduction.md#1-movielens)



### Amazon

[Amazon](http://jmcauley.ucsd.edu/data/amazon/)提供了商品数据集，该数据集包含亚马逊的产品评论和元数据，包括1996年5月至2014年7月期间的1.428亿评论。它包括很多子数据集，如：Book、Electronics、Movies and TV等，实验中我们主要使用**Electronics子数据集**。

Amazon-Electronics数据集的具体介绍与处理：[传送门](~document/Dataset%20Introduction.md#2-amazon)



### Criteo

Criteo广告数据集是一个经典的用来预测广告点击率的数据集。2014年，由全球知名广告公司Criteo赞助举办[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)比赛。但比赛过去太久，Kaggle已不提供数据集。现有三种方式获得数据集或其样本：

1. [Criteo_sample.txt](https://github.com/shenweichen/DeepCTR/blob/master/examples/criteo_sample.txt)：包含在DeepCTR中，用于测试模型是否正确，不过数据量太少；
2. [kaggle Criteo](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz)：训练集（10.38G）、测试集（1.35G）;（实验大部分都是使用该数据集）
3. [Criteo 1TB](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)：可以根据需要下载完整的日志数据集；

Criteo数据集的具体介绍与处理：[传送门](~document/Dataset%20Introduction.md#3-criteo)



## 复现论文

### 1. 传统推荐模型



|                        Paper\|Model                         |        Published in        |            Author            |
| :---------------------------------------------------------: | :------------------------: | :--------------------------: |
| Matrix Factorization Techniques for Recommender Systems\|MF | IEEE Computer Society,2009 |    Koren\|Yahoo Research     |
|                 Factorization Machines\|FM                  |         ICDM, 2010         |        Steffen Rendle        |
| Field-aware Factorization Machines for CTR Prediction｜FFM  |        RecSys, 2016        | Yuchin Juan｜Criteo Research |



### 2. 基于神经网络的模型



|                         Paper｜Model                         | Published in |                            Author                            |
| :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: |
|      Wide & Deep Learning for Recommender Systems｜WDL       |  DLRS, 2016  |                         Google Inc.                          |
| Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features\|Deep Crossing |  KDD, 2016   |                      Microsoft Research                      |
| Product-based Neural Networks for User Response Prediction\|PNN |  ICDM, 2016  |                Shanghai Jiao Tong University                 |
|      Deep & Cross Network for Ad Click Predictions｜DCN      | ADKDD, 2017  |               Stanford University｜Google Inc.               |
| Neural Factorization Machines for Sparse Predictive Analytics\|NFM | SIGIR, 2017  |                         Xiangnan He                          |
|      Neural network-based Collaborative Filtering\|NCF       |  WWW, 2017   |                         Xiangnan He                          |
| Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks\|AFM | IJCAI, 2017  |    Zhejiang University\|National University of Singapore     |
| DeepFM: A Factorization-Machine based Neural Network for CTR Prediction\|DeepFM | IJCAI, 2017  | Harbin Institute of Technology\|Noah’s Ark Research Lab, Huawei |
| xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems\|xDeepFM |  KDD, 2018   |        University of Science and Technology of China         |



### 3. 序列模型



| Paper｜Model                                                 | Published in | Author        |
| ------------------------------------------------------------ | ------------ | ------------- |
| Deep Interest Network for Click-Through Rate Prediction\|DIN | KDD, 2018    | Alibaba Group |





## 联系方式

1、对于项目有任何建议或问题，可以在`Issue`留言，或者可以添加作者微信`zgzjhzgzy`。

2、作者有一个自己的公众号：`推荐算法的小齿轮`，如果喜欢里面的内容，不妨点个关注。

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="30%"/></div>

