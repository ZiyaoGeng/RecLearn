<div>
  <img src='https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_0.png' width='40%'/>
</div>

## 前言

<p align="left">
  <img src='https://img.shields.io/badge/python-3.7-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
  <img src='https://img.shields.io/badge/Tensorflow-2.0-brightgreen'>
</p>  

本人在读研一，推荐系统方向，和各位一样，在推荐算法这条路愈走愈远，无法自拔。开源项目`Recommended System with TF2.0`主要是对**阅读过的部分推荐系统、CTR预估论文进行复现**。建立的**原因**有三个：

1. 理论和实践似乎有很大的间隔，学术界与工业界的差距更是如此；
2. 更好的理解论文的核心内容，增强自己的工程能力；
3. 很多论文给出的开源代码都是TF1.x，因此想要用更简单的TF2.0进行复现；

当然也看过一些知名的开源项目，如[DeepCTR](https://github.com/shenweichen/DeepCTR)等，不过对自己目前的水平来说，只适合拿来参考。

**项目特点：**

- 使用Tensorflow2.0进行复现；
- 每个模型都是相互独立的，不存在依赖关系；
- 模型基本按照论文进行构建，实验尽量使用论文给出的的公共数据集。如果论文给出github代码，会进行参考；
- 对于[实验数据集](#数据集介绍)有专门详细的介绍；
- 代码源文件参数、函数命名规范，并且带有标准的注释；
- 每个模型会有专门的代码文档（`.md文件`）或者其他方式进行解释；

## 目录

目前**复现的模型**有（按复现时间进行排序）：

1. [NCF](#1-neural-network-based-collaborative-filteringncf)
2. [DIN](#2-deep-interest-network-for-click-through-rate-predictiondin)
3. [Wide&Deep](#3-wide--deep-learning-for-recommender-systems)
4. [DCN](#4-deep--cross-network-for-ad-click-predictions)
5. [PNN](#5product-based-neural-networks-for-user-response-prediction)
6. [Deep Crossing](#6-deep-crossing-web-scale-modeling-without-manually-crafted-combinatorial-features)
7. [DeepFM](#7-deepfm-a-factorization-machine-based-neural-network-for-ctr-prediction)
8. [NFM](#8-neural-factorization-machines-for-sparse-predictive-analytics)
9. [AFM](#9-attentional-factorization-machines-learning-the-weight-of-feature-interactions-via-attention-networks)
10. [xDeepFM](#10-xdeepfm-combining-explicit-and-implicit-feature-interactions-for-recommender-systems)

## 更新

2020.08.21：xDeepFM模型；

2020.08.03：AFM模型；

2020.08.02：NFM模型；

2020.07.31：DeepFM模型；

2020.07.29：[PNN代码文档](./PNN/PNN_document.md)更新；

2020.07.28：更改ReadMe介绍；

2020.07.27：Deep Crossing模型；

2020.07.20：PNN模型；

2020.07.14：DCN模型；

2020.07.10：Wide&Deep模型；

2020.05.26：DIN模型；

2020.03.27：NCF模型；



## 数据集介绍

目前实验使用的数据集主要有三个：Movielens、Amazon、Criteo。

### Movielens

[MovieLens](https://grouplens.org/datasets/movielens/)是历史最悠久的推荐系统数据集，主要分为：ml-100k（1998年）、ml-1m（2003年）、ml-10m（2009年）、ml-20m（2015年）、ml-25m（2019年）。实验中主要使用ml-1m数据集。

已处理过的数据集：[ml-1m](https://github.com/hexiangnan/neural_collaborative_filtering)

ml-1m数据集的具体介绍与处理：[传送门](./Dataset%20Introduction.md#1-movielens)



### Amazon

[Amazon](http://jmcauley.ucsd.edu/data/amazon/)提供了商品数据集，该数据集包含亚马逊的产品评论和元数据，包括1996年5月至2014年7月期间的1.428亿评论。它包括很多子数据集，如：Book、Electronics、Movies and TV等，实验中我们主要使用**Electronics子数据集**。

Amazon-Electronics数据集的具体介绍与处理：[传送门](./Dataset%20Introduction.md#2-amazon)



### Criteo

Criteo广告数据集是一个经典的用来预测广告点击率的数据集。2014年，由全球知名广告公司Criteo赞助举办[Display Advertising Challenge](https://www.kaggle.com/c/criteo-display-ad-challenge)比赛。但比赛过去太久，Kaggle已不提供数据集。现有三种方式获得数据集或其样本：

1. [Criteo_sample.txt](https://github.com/shenweichen/DeepCTR/blob/master/examples/criteo_sample.txt)：包含在DeepCTR中，用于测试模型是否正确，不过数据量太少；
2. [kaggle Criteo](https://s3-eu-west-1.amazonaws.com/kaggle-display-advertising-challenge-dataset/dac.tar.gz)：训练集（10.38G）、测试集（1.35G）;（实验大部分都是使用该数据集）
3. [Criteo 1TB](https://labs.criteo.com/2013/12/download-terabyte-click-logs/)：可以根据需要下载完整的日志数据集；

Criteo数据集的具体介绍与处理：[传送门](./Dataset%20Introduction.md#3-criteo)



## 复现论文

### 1. Neural network-based Collaborative Filtering（NCF）

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_1.png" width="40%"/></div>

**数据集：**

Movielens、Pinterest

**代码解析：**

**原文开源代码：**

https://github.com/hexiangnan/neural_collaborative_filtering

**原文笔记：**

**目录：**[返回目录  ](#目录)



### 2. Deep Interest Network for Click-Through Rate Prediction(DIN)

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_2.png" width="40%;" style="float:center"/></div>

**数据集：**

Amazon数据集中Electronics子集。

**代码解析：**

https://zhuanlan.zhihu.com/p/144153291

**参考原文开源代码地址：**

https://github.com/zhougr1993/DeepInterestNetwork

**原文笔记：**

https://mp.weixin.qq.com/s/uIs_FpeowSEpP5fkVDq1Nw

**目录：**[返回目录  ](#目录)

  

### 3. Wide & Deep Learning for Recommender Systems

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_3.png" width="60%;" style="float:center"/></div>

**数据集：**

由于原文没有给出公开数据集，所以在此我们使用Amazon Dataset中的Electronics子集，由于数据集的原因，模型可能与原文的有所出入，但整体思想还是不变的。

**代码解析：**

**原文笔记：**

https://mp.weixin.qq.com/s/LRghf8mj1hjUYri_m3AzBg

  **目录：**[返回目录  ](#目录)



### 4. Deep & Cross Network for Ad Click Predictions

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_4.png" width="40%;" style="float:center"/></div>

**数据集：**

Criteo

**代码解析：**

**原文笔记：**

https://mp.weixin.qq.com/s/DkoaMaXhlgQv1NhZHF-7og

  **目录：**[返回目录  ](#目录)



### 5.Product-based Neural Networks for User Response Prediction

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_5.png" width="40%;" style="float:center"/></div>

**数据集：**

Criteo

**代码解析：**

[PNN代码文档](./PNN/PNN_document.md)

**原文笔记：**

https://mp.weixin.qq.com/s/GMQd5RTmGPuxbokoHZs3eg

**目录：**[返回目录  ](#目录)




### 6. Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_6.png" width="40%;" style="float:center"/></div>

**数据集：**

Crieto

**代码解析：**

[Deep Crossing代码文档](./~document/Deep_Crossing_document.md)

**原文笔记：**

https://mp.weixin.qq.com/s/WXnvkoRFxwFpflStAuW7kQ

**目录：**[返回目录  ](#目录)



### 7. DeepFM: A Factorization-Machine based Neural Network for CTR Prediction

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_7.png" width="40%;" style="float:center"/></div>

**数据集：**

Crieto

**代码解析：**

[DeepFM代码文档](./~document/DeepFM_document.md)

**原文笔记：**

https://mp.weixin.qq.com/s/bxYag1GcJABkwwz0NmLI5g

**目录：**[返回目录  ](#目录)



### 8. Neural Factorization Machines for Sparse Predictive Analytics

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_8.png" width="40%;" style="float:center"/></div>

**数据集：**

Crieto

**代码解析：**

[NFM代码文档](./~document/NFM_document.md)

**原文笔记：**

https://mp.weixin.qq.com/s/1en7EyP3C2TP3-d4Ha0rSQ

**目录：**[返回目录  ](#目录)



### 9. Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_9.png" width="50%;" style="float:center"/></div>

**数据集：**

Crieto

**原文开源代码：**

https://github.com/hexiangnan/attentional_factorization_machine

**代码解析：**

[AFM代码文档](./~document/AFM_document.md)

**原文笔记：**

https://mp.weixin.qq.com/s/hPCS9Dw2vT2pwdWwPo0EJg

**目录：**[返回目录  ](#目录)



### 10. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

**模型：**

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_10.png" width="50%;" style="float:center"/></div>

**数据集：**

Criteo

**原文开源代码：**

https://github.com/Leavingseason/xDeepFM

**原文笔记：**

https://mp.weixin.qq.com/s/TohOmVpQzNlA3vXv0gpobg

**目录：**[返回目录  ](#目录)

## 附

公众号：**潜心的Python小屋**，欢迎大家关注。

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_weixin.png" width="30%;" style="float:center"/></div>
