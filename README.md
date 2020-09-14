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

开源项目`Recommender System with TF2.0`主要是对阅读过的部分推荐系统、CTR预估论文进行复现，包括**传统模型**（MF、FM、FFM等）、**神经网络模型**（WDL、DCN等）以及**序列模型**（DIN）。

**建立原因：**

1. 理论和实践似乎有很大的间隔，学术界与工业界的差距更是如此；
2. 更好的理解论文的核心内容，增强自己的工程能力；
3. 很多论文给出的开源代码都是TF1.x，因此想要用更简单的TF2.0进行复现；

**项目特点：**

- 使用Tensorflow2.0进行复现；
- 每个模型都是相互独立的，不存在依赖关系；
- 模型基本按照论文进行构建，实验尽量使用论文给出的的公共数据集；
- 具有【[Wiki](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki)】，对于模型、实验数据集有详细的介绍和链接；
- 代码源文件参数、函数命名规范，并且带有标准的注释；



&nbsp;

## 实验

1、通过git命令`git clone https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0.git`或者直接下载；

2、需要环境Python3.7，Tensorflow2.0；

3、根据自己数据集的位置，合理更改所需模型文件内`train.py`的`file`路径；

4、设置`超参数`，直接运行即可；



&nbsp;

## 复现论文

### 1. 传统推荐模型



|                         Paper\|Model                         |        Published in        |            Author            |
| :----------------------------------------------------------: | :------------------------: | :--------------------------: |
| Matrix Factorization Techniques for Recommender Systems\|**MF** | IEEE Computer Society,2009 |    Koren\|Yahoo Research     |
|                Factorization Machines\|**FM**                |         ICDM, 2010         |        Steffen Rendle        |
| Field-aware Factorization Machines for CTR Prediction｜**FFM** |        RecSys, 2016        | Yuchin Juan｜Criteo Research |

&nbsp;

### 2. 基于神经网络的模型



|                         Paper｜Model                         | Published in |                            Author                            |
| :----------------------------------------------------------: | :----------: | :----------------------------------------------------------: |
| Wide & Deep Learning for Recommender Systems｜**[WDL](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Wide-&-Deep-Learning)** |  DLRS, 2016  |                         Google Inc.                          |
| Deep Crossing: Web-Scale Modeling without Manually Crafted Combinatorial Features\|**[Deep Crossing](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Deep-Crossing)** |  KDD, 2016   |                      Microsoft Research                      |
| Product-based Neural Networks for User Response Prediction\|[**PNN**](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Product-based-Neural-Networks) |  ICDM, 2016  |                Shanghai Jiao Tong University                 |
| Deep & Cross Network for Ad Click Predictions｜[**DCN**](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Deep-&-Cross-Network) | ADKDD, 2017  |               Stanford University｜Google Inc.               |
| Neural Factorization Machines for Sparse Predictive Analytics\|**[NFM](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/wiki/Neural-Factorization-Machines)** | SIGIR, 2017  |                         Xiangnan He                          |
|    Neural network-based Collaborative Filtering\|**NCF**     |  WWW, 2017   |                         Xiangnan He                          |
| Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks\|**AFM** | IJCAI, 2017  |    Zhejiang University\|National University of Singapore     |
| DeepFM: A Factorization-Machine based Neural Network for CTR Prediction\|**DeepFM** | IJCAI, 2017  | Harbin Institute of Technology\|Noah’s Ark Research Lab, Huawei |
| xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems\|**xDeepFM** |  KDD, 2018   |        University of Science and Technology of China         |

&nbsp;

### 3. 序列模型

| Paper｜Model                                                 | Published in | Author        |
| ------------------------------------------------------------ | ------------ | ------------- |
| Deep Interest Network for Click-Through Rate Prediction\|**DIN** | KDD, 2018    | Alibaba Group |
| Self-Attentive Sequential Recommendation｜**SASRec**         | ICDM, 2018   | UCSD          |



&nbsp;

## 联系方式

1、对于项目有任何建议或问题，可以在`Issue`留言，或者可以添加作者微信`zgzjhzgzy`。

2、作者有一个自己的公众号：**推荐算法的小齿轮**，如果喜欢里面的内容，不妨点个关注。

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="30%"/></div>

