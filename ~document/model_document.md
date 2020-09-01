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
11. [FM](#11-factorization-machines)
12. [FFM](#12-field-aware-factorization-machines-for-ctr-prediction)
13. [MF](#13-matrix-factorization-techniques-for-recommender-systems)

## 

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



### 11. Factorization Machines

**数据集：**

Criteo

**原文笔记：**

https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html

**目录：**[返回目录  ](#目录)



### 12. Field-aware Factorization Machines for CTR Prediction

**数据集：**

Criteo

**C++包：**

https://github.com/ycjuan/libffm

【注】FFM复现只是为了让自己更清楚其构造，但是真正在场景或比赛中应用的话，还是调取上述包。因为自己不会优化，两个for循环太慢了。

**原文笔记**：

https://tech.meituan.com/2016/03/03/deep-understanding-of-ffm-principles-and-practices.html

**目录：**[返回目录  ](#目录)



### 13. Matrix Factorization Techniques for Recommender Systems

**数据集：**

ml-1m

**目录：**[返回目录  ](#目录)