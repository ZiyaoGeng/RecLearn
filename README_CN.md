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

RecLearn（Recommender Learning）对`Recommender System with TF2.0`中 [master](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/master) 分支的内容进行了归纳、整理，是一个基于Python和Tensorflow2.x开发的推荐学习框架，适合学生、初学者研究使用。**当然如果你更习惯master分支中的内容，可以直接clone整个包的内容，在example中运行一些算法，并且也能对model、layer中的内容进行更新和修改**。实现的推荐算法按照工业界的两个应用阶段进行分类：

- matching recommendation stage
- ranking  recommendeation stage



## 安装

### Package

RecLearn已经上传在pypi上，可以使用`pip`进行安装：

```shell
pip install reclearn
```

所依赖的环境：

- python3.7+
- Tensorflow2.7+
- sklearn



### Local

也可以直接clone Reclearn到本地：

```shell
git clone -b reclearn git@github.com:ZiyaoGeng/Recommender-System-with-TF2.0.git
```



## 快速开始

在`example`中，给出了每一个推荐模型的demo。

### Matching

**1、分割数据集**

给定数据集的路径：

```python
file_path = 'data/ml-1m/ratings.dat'
```

划分当前数据集为训练集、验证集、测试集。如果你使用了`movielens-1m`、`Amazon-Beauty`、`Amazon-Games`、`STEAM`数据集的话，也可以直接调用Reclearn中`data/datasets/*`的方法，在数据集的目录中完成划分（**并对用户、物品的ID从1开始重新进行映射**）：

```python
train_path, val_path, test_path, meta_path = ml.split_seq_data(file_path=file_path)
```

其中`meta_path`为元文件的路径，元文件保存了用户、物品索引的最大值。

**2、建立特征列**

特征列是一个字典，例如：

```python
fea_cols = {
        'item': sparseFeature('item', max_item_num + 1, embed_dim),
        'user': sparseFeature('user', max_user_num + 1, embed_dim)
    }
```

其中`sparseFeature`是对特征的描述，包括名称、最大值以及Embedding的维度。

**3、加载数据**

完成对训练集、验证集、测试集的读取，并且对每一个正样本分别生成若干个负样本，数据的格式为字典：

```
data = {'pos_item':, 'neg_item': , ['user': , 'click_seq': ,...]}
```

如果你构建的模型为序列推荐模型，需要引入点击序列，使用了用户ID特征，也需要引入。对于上述4个数据集，Reclearn提供了加载数据的方法：

```python
# seq rec model, and use the user feature.
train_data = ml.load_seq_data(train_path, "train", seq_len, neg_num, max_item_num, contain_user=True)
# general model
train_data = ml.load_data(train_path, neg_num, max_item_num)
```

**4、构建模型、编译**

选择或构建你需要的模型，设置好超参数，并进行编译。以`AttRec`为例：

```python
model = AttRec(fea_cols, **model_params)
model.compile(optimizer=Adam(learning_rate=learning_rate))
```

如果你对模型的结构存在问题的话，编译之后可以调用`summary`方法打印查看：

```python
model.summary()
```

**5、学习以及预测**。

```python
for epoch in range(1, epochs + 1):
    t1 = time()
    model.fit(
        x=train_data,
        epochs=1,
        validation_data=val_data,
        batch_size=batch_size
    )
    t2 = time()
    eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], k, batch_size)
    print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f'
          % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))
```



### Ranking

参考`example/r_deepfm_demo.py`文件

**1、分割数据集**

**2、建立特征映射**

**3、加载测试集**

**4、构建模型**

**5、迭代训练，并验证**



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



## 讨论

对于项目有任何建议或问题，可以在`Issue`留言，或者发邮件至`zggzy1996@163.com`。