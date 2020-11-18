<div>
  <img src='https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/logo.jpg' width='36%'/>
</div>

## 前言

<p align="left">
  <img src='https://img.shields.io/badge/python-3.7-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
  <img src='https://img.shields.io/badge/sklearn-0.23.2-brightgreen'>
  <img src='https://img.shields.io/badge/Tensorflow-2.0-brightgreen'>
</p>  

开源项目`Recommender System with TF2.0`主要是对阅读过的部分推荐系统、CTR预估论文进行复现，包括**Match（召回）**（NCF、SASRec、STAMP等）、**Rank（粗排）**（WDL、DCN等）。

**建立原因：**

1. 理论和实践似乎有很大的间隔，学术界与工业界的差距更是如此；
2. 更好的理解论文的核心内容，增强自己的工程能力；
3. 很多论文给出的开源代码都是TF1.x，因此想要用更简单的TF2.x进行复现；

**项目特点：**

- 使用Tensorflow2.x进行复现；
- 每个模型都是相互独立的，不存在依赖关系；
- 模型基本按照论文进行构建，实验尽量使用论文给出的的公共数据集；
- 模型都附有`README.md`，对于模型的训练使用有详细的介绍；
- 代码源文件参数、函数命名规范，并且带有标准的注释；



## 重要更新

1. 【2020.11.18】在Top-K模型中，不再考虑`dense_inputs`、`sparse_inputs`，并且`user_inputs`和`seq_inputs`不考虑多个类别，只将`id`特征作为输入（降低了模型的可扩展性，但是提高了模型的可读性）；
2. 【2020.11.18】BPR、SASRec模型进行了更新，加入了实验结果；



&nbsp;

## 实验

1、通过git命令`git clone https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0.git`或者直接下载；

2、实验环境：Python3.7，Tensorflow2.0-CPU/GPU；

3、根据自己数据集的位置，合理更改所需模型文件内`train.py`的`file`路径；

4、设置`超参数`，直接运行即可；



&nbsp;

## 数据集

-   [Movielens](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Movielens)
-   [Amazon Dataset](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Amazon-Dataset)
-   [Criteo Dataset](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Criteo-Dataset)
- [Diginetica Dataset](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Diginetica-Dataset)



&nbsp;

## 复现论文

### 1. 召回模型（Top-K推荐）



|                         Paper\|Model                         |        Published in        |        Author         |
| :----------------------------------------------------------: | :------------------------: | :-------------------: |
| Matrix Factorization Techniques for Recommender Systems\|**[MF](MF)** | IEEE Computer Society,2009 | Koren\|Yahoo Research |
| BPR: Bayesian Personalized Ranking from Implicit Feedback\|**[MF-BPR](BPR)** |         UAI, 2009          |     Steﬀen Rendle     |
| Neural network-based Collaborative Filtering\|[**NCF**](NCF) |         WWW, 2017          |      Xiangnan He      |
| Self-Attentive Sequential Recommendation｜[**SASRec**](SASRec) |         ICDM, 2018         |         UCSD          |
| STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation\| **[STAMP](STAMP)** |         KDD, 2018          |       Qiao Liu        |

&nbsp;

### 2. 粗排模型（CTR预估）

|                         Paper｜Model                         | Published in |                            Author                            |
| :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: |
|             Factorization Machines\|**[FM](FM)**             |  ICDM, 2010  |                        Steffen Rendle                        |
| Field-aware Factorization Machines for CTR Prediction｜[**FFM**](FFM) | RecSys, 2016 |                 Yuchin Juan｜Criteo Research                 |
| Wide & Deep Learning for Recommender Systems｜[**WDL**](WDL) |  DLRS, 2016  |                         Google Inc.                          |
| Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features\|**[Deep Crossing](Deep_Crossing)** |  KDD, 2016   |                      Microsoft Research                      |
| Product-based Neural Networks for User Response Prediction\|[**PNN**](PNN) |  ICDM, 2016  |                Shanghai Jiao Tong University                 |
| Deep & Cross Network for Ad Click Predictions｜[**DCN**](DCN) | ADKDD, 2017  |               Stanford University｜Google Inc.               |
| Neural Factorization Machines for Sparse Predictive Analytics\|**[NFM](NFM)** | SIGIR, 2017  |                         Xiangnan He                          |
| Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks\|**[AFM](AFM)** | IJCAI, 2017  |    Zhejiang University\|National University of Singapore     |
| DeepFM: A Factorization-Machine based Neural Network for CTR Prediction\|**[DeepFM](DeepFM)** | IJCAI, 2017  | Harbin Institute of Technology\|Noah’s Ark Research Lab, Huawei |
| xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems\|**[xDeepFM](xDeepFM)** |  KDD, 2018   |        University of Science and Technology of China         |
| Deep Interest Network for Click-Through Rate Prediction\|**[DIN](DIN)** |  KDD, 2018   |                        Alibaba Group                         |

&nbsp;

## 致谢

项目中会存在一些代码Bug，感谢以下朋友指出问题：

1. [wangzhe258369](https://github.com/wangzhe258369)：指出在DIN模型中`tf.keras.layers.BatchNormalization`默认行为是`training=False`，此时不会去更新BN中的moving_mean和moving_variance变量。但是重新修改了DIN模型代码内容时，再仔细查找了资料，[发现](https://www.it610.com/article/1276108622954250240.htm)：

   > 如果使用模型调用fit()的话，是可以不给的（官方推荐是不给），因为在fit()的时候，模型会自己根据相应的阶段（是train阶段还是inference阶段）决定training值，这是由learning——phase机制实现的。

2. **[boluochuile](https://github.com/boluochuile)**：发现SASRec模型训练出错，原因是验证集必须使用`tuple`的方式，已更正；

3. **[boluochuile](https://github.com/boluochuile)**：指出SASRec模型中label赋值的问题，

   ```python
   data_df.loc[data_df.label >= 2, 'label'] = 1
   data_df.loc[data_df.label < 2, 'label'] = 0
   ```

   应该颠倒，已更正；
   
4. **[dominic-z](https://github.com/dominic-z)**：指出DIN中Attention的mask问题，更改为从`seq_inputs`中得到mask，因为采用的是0填充（这里与重写之前的代码不同，之前是在每个mini-batch中选择最大的长度作为序列长度，不会存在序列过长被切割的问题，而现在为了方便，采用最普遍`padding`的方法）

5. **[dominic-z](https://github.com/dominic-z)**：指出DIN训练中`seq_inputs`shape与model不匹配的问题，已更正，应该是`(batch_size,  maxlen, behavior_num)`，model相关内容进行更改，另外对于行为数量，之前的名称`seq_len`有歧义，改为`behavior_num`；**添加了重写之前的代码，在DIN/old目录下**

&nbsp;

## 联系方式

1、对于项目有任何建议或问题，可以在`Issue`留言，或者可以添加作者微信`zgzjhzgzy`。

2、作者有一个自己的公众号：**推荐算法的小齿轮**，如果喜欢里面的内容，不妨点个关注。

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="30%"/></div>

