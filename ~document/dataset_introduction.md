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

