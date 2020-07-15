# Recommended-System with TensorFlow 2.0

### 前言

本人在读研一，推荐系统方向。看过一些比较知名的推荐系统、CTR预估论文。开【Recommended-System with TensorFlow 2.0】的原因有三个：

1. 论文只看理论感觉有些地方简单，但是实践起来却比较困难；
2. 为了更好的理解论文，增强自己的工程能力；
3. 很多论文给出的复现代码都是TF1.x，对于使用TF2.0的我来说，很难理解【TF1.x学习不系统】；

所以想开一个TF2.0的Project，来对部分论文进行实验的复现。当然也看过一些知名的开源项目，如deepcrt等，不过发现对自己来说，只适合拿来参考【代码水平太高了，写不出来】。

关于【Recommended-System with TensorFlow 2.0】，模型基本按照论文进行构建，实验尽量使用论文给出的的公共数据集。如果论文给出github代码，会进行参考。

目前**复现的模型**有：NCF、DIN、Wide&Deep、DCN。

**快速导航：**

1. [NCF](https://github.com/BlackSpaceGZY/Recommended-System#1. Neural network-based Collaborative Filtering（NCF）)
2. [DIN](https://github.com/BlackSpaceGZY/Recommended-System#2. Deep Interest Network for Click-Through Rate Prediction(DIN))
3. [Wide&Deep](https://github.com/BlackSpaceGZY/Recommended-System#3. Wide & Deep Learning for Recommender Systems)
4. [DCN](https://github.com/BlackSpaceGZY/Recommended-System#4-deep--cross-network-for-ad-click-predictions)

  

### 实验环境

Python 3.7；

Tensorflow 2.0-CPU；  

  

### 复现论文

#### 1. Neural network-based Collaborative Filtering（NCF）

**模型：**

<img src="images/1.png" style="zoom:50%;" />

**数据集：**

Movielens、Pinterest

**代码：**

- Data：数据集
- Pretrain：预训练保存的模型；
- Save：模型保存；
- configs.py：参数设置，对应模型有对应的参数；
- DataSet.py：构造所需要的数据集，得到负样本集合；
- evaluate.py：评估函数；
- GMF.py：模型；
- MLP.py：模型；
- NeuMF.py：模型；
- utils.py：获得训练样本，加载预训练模型；

**参考原文开源代码地址：**

https://github.com/hexiangnan/neural_collaborative_filtering

**原文地址：**

https://arxiv.org/pdf/1708.05031.pdf?source=post_page---------------------------

**原文笔记：**

  

#### 2. Deep Interest Network for Click-Through Rate Prediction(DIN)

**模型：**

<img src="images/2.png" style="zoom:50%;" />

**数据集：**

[Amazon](http://jmcauley.ucsd.edu/data/amazon/)数据集中Electronics子集，下载并解压【或手动下载】：

```python
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
```

其中`reviews_Electronics_5.json`为用户的行为数据，`meta_Electronics`为广告的元数据。

**代码：**

- datasset：处理过的数据集，dataset.kpl；
- logs：TensorBoard所保存的日志；
- raw_data：原数据存放地址；
- save：模型保存；
- utils：处理数据
  - 1_convert_pd.py
  - 2_remap_id.py
- build_dataset.py：构建数据集；
- dice.py：Dice实现；
- model.py：模型；
- train.py：训练；

**参考原文开源代码地址：**

https://github.com/zhougr1993/DeepInterestNetwork

**原文地址：**

https://arxiv.org/pdf/1706.06978.pdf

**原文笔记：**

https://mp.weixin.qq.com/s/uIs_FpeowSEpP5fkVDq1Nw

  

#### 3. Wide & Deep Learning for Recommender Systems

**模型：**

![](images/3.png)

对于Wide&Deep模型来说，Tensorflow中有内置的模型。

**数据集：**

由于原文没有给出公开数据集，所以在此我们使用Amazon Dataset中的Electronics子集，由于数据集的原因，模型可能与原文的有所出入，但整体思想还是不变的。

**代码：**

- logs：TensorBoard保存日志；
- save：模型保存；
- model.py：模型；
- train.py：训练

注：数据集使用了DIN中的构造，所以直接调用了。

**原文地址：**

[https://arxiv.org/pdf/1606.07792.pdf%29/](https://arxiv.org/pdf/1606.07792.pdf)/)

**原文笔记：**

https://mp.weixin.qq.com/s/LRghf8mj1hjUYri_m3AzBg

  

#### 4. Deep & Cross Network for Ad Click Predictions

**模型：**

<img src="images/4.png" style="zoom:50%;" />

**数据集：**

Criteo Kaggle比赛数据集。

注：由于Kaggle数据已经不公开，且只是为了测试，所以使用了一个小样本数据集【参考了deepctr】，如果想在原数据集上进行实验，可去寻找相关资源。

**代码：**

- dataset：数据集；
- log：TensorBoard保存日志；
- save：模型保存；
- model.py：模型；
- train.py：训练；
- utils.py：数据处理；

**原文地址：**

https://arxiv.org/pdf/1708.05123.pdf

**原文笔记：**

  

### 附

公众号：潜心的Python小屋，欢迎大家关注。

<img src="images/0.png" style="zoom:50%;" />