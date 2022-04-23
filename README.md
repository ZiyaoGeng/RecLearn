<div>
  <img src='https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/logo.jpg' width='36%'/>
</div>

## RecLearn

<p align="left">
  <img src='https://img.shields.io/badge/python-3.8+-blue'>
  <img src='https://img.shields.io/badge/Tensorflow-2.5+-blue'>
  <img src='https://img.shields.io/badge/License-MIT-blue'>
  <img src='https://img.shields.io/badge/NumPy-1.17-brightgreen'>
  <img src='https://img.shields.io/badge/pandas-1.0.5-brightgreen'>
  <img src='https://img.shields.io/badge/sklearn-0.23.2-brightgreen'>
</p> 

[简体中文](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/blob/reclearn/README_CN.md) | [English](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/reclearn)

RecLearn (Recommender Learning)  which summarizes the contents of the [master](https://github.com/ZiyaoGeng/RecLearn/tree/master) branch in  `Recommender System with TF2.0 `  is a recommended learning framework based on Python and TensorFlow2.x for students and beginners. **Of course, if you are more comfortable with the master branch, you can clone the entire package, run some algorithms in example, and also update and modify the content of model and layer**. The implemented recommendation algorithms are classified according to two application stages in the industry:

- matching recommendation stage (Top-k Recmmendation)
- ranking  recommendeation stage (CTR predict model)



## Update

**04/23/2022**: update all matching model.



## Installation

### Package

RecLearn is on PyPI, so you can use pip to install it.

```
pip install reclearn
```

dependent environment：

- python3.8+
- Tensorflow2.5-GPU+/Tensorflow2.5-CPU+
- sklearn0.23+

### Local

Clone Reclearn to local:

```shell
git clone -b reclearn git@github.com:ZiyaoGeng/RecLearn.git
```



## Quick Start

In [example](https://github.com/ZiyaoGeng/Recommender-System-with-TF2.0/tree/reclearn/example), we have given a demo of each of the recommended models.

### Matching

**1. Divide the dataset.**

Set the path of the raw dataset:

```python
file_path = 'data/ml-1m/ratings.dat'
```

Please divide the current dataset into training dataset, validation dataset and test dataset. If you use `movielens-1m`, `Amazon-Beauty`, `Amazon-Games` and `STEAM`, you can call method `data/datasets/*` of RecLearn directly:

```python
train_path, val_path, test_path, meta_path = ml.split_seq_data(file_path=file_path)
```

`meta_path` indicates the path of the metafile, which stores the maximum number of user and item indexes.

**2. Load the dataset.**

Complete the loading of training dataset, validation dataset and test dataset, and generate several negative samples (random sampling) for each positive sample. The format of data is dictionary:

```python
data = {'pos_item':, 'neg_item': , ['user': , 'click_seq': ,...]}
```

If you're building a sequential recommendation model, you need to introduce click sequences. Reclearn provides methods for loading the data for the above four datasets:

```python
# general recommendation model
train_data = ml.load_data(train_path, neg_num, max_item_num)
# sequence recommendation model, and use the user feature.
train_data = ml.load_seq_data(train_path, "train", seq_len, neg_num, max_item_num, contain_user=True)
```

**3. Set hyper-parameters.**

The model needs to specify the required hyperparameters. Now, we take `BPR` model as an example:

```python
model_params = {
        'user_num': max_user_num + 1,
        'item_num': max_item_num + 1,
        'embed_dim': FLAGS.embed_dim,
        'use_l2norm': FLAGS.use_l2norm,
        'embed_reg': FLAGS.embed_reg
    }
```

**4. Build and compile the model.**

Select or build the model you need and compile it. Take 'BPR' as an example:

```python
model = BPR(**model_params)
model.compile(optimizer=Adam(learning_rate=FLAGS.learning_rate))
```

If you have problems with the structure of the model, you can call the summary method after compilation to print it out:

```python
model.summary()
```

**5. Learn the model and predict test dataset.**

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

Waiting......



## Results

The experimental environment designed by Reclearn is different from that of some papers, so there may be some deviation in the results. Please refer to [Experiement](./docs/experiment.md) for details.

### Matching

<table style="text-align:center;margin:auto">
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
  <tr><td>BPR</td><td>0.5768</td><td>0.2392</td><td>0.3016</td><td>0.3708</td><td>0.2108</td><td>0.2485</td><td>0.7728</td><td>0.4220</td><td>0.5054</td></tr>
  <tr><td>NCF</td><td>0.5834</td><td>0.2219</td><td>0.3060</td><td>0.5448</td><td>0.2831</td><td>0.3451</td><td>0.7768</td><td>0.4273</td><td>0.5103</td></tr>
  <tr><td>DSSM</td><td>0.5498</td><td>0.2148</td><td>0.2929</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>YoutubeDNN</td><td>0.6737</td><td>0.3414</td><td>0.4201</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>GRU4Rec</td><td>0.7969</td><td>0.4698</td><td>0.5483</td><td>0.5211</td><td>0.2724</td><td>0.3312</td><td>0.8501</td><td>0.5486</td><td>0.6209</td></tr>
  <tr><td>Caser</td><td>0.7916</td><td>0.4450</td><td>0.5280</td><td>0.5487</td><td>0.2884</td><td>0.3501</td><td>0.8275</td><td>0.5064</td><td>0.5832</td></tr>
  <tr><td>SASRec</td><td>0.8103</td><td>0.4812</td><td>0.5605</td><td>0.5230</td><td>0.2781</td><td>0.3355</td><td>0.8606</td><td>0.5669</td><td>0.6374</td></tr>
  <tr><td>AttRec</td><td>0.7873</td><td>0.4578</td><td>0.5363</td><td>0.4995</td><td>0.2695</td><td>0.3229</td><td>-</td><td>-</td><td>-</td></tr>
  <tr><td>FISSA</td><td>0.8106</td><td>0.4953</td><td>0.5713</td><td>0.5431</td><td>0.2851</td><td>0.3462</td><td>0.8635</td><td>0.5682</td><td>0.6391</td></tr>
</table>



### Ranking

<table style="text-align:center;margin:auto">
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

|                         Paper\|Model                         |  Published   |     Author     |
| :----------------------------------------------------------: | :----------: | :------------: |
| BPR: Bayesian Personalized Ranking from Implicit Feedback\|**MF-BPR** |  UAI, 2009   | Steﬀen Rendle  |
|    Neural network-based Collaborative Filtering\|**NCF**     |  WWW, 2017   |  Xiangnan He   |
| Learning Deep Structured Semantic Models for Web Search using Clickthrough Data\|**DSSM** |  CIKM, 2013  |  Po-Sen Huang  |
| Deep Neural Networks for YouTube Recommendations\| **YoutubeDNN** | RecSys, 2016 | Paul Covington |
| Session-based Recommendations with Recurrent Neural Networks\|**GUR4Rec** |  ICLR, 2016  | Balázs Hidasi  |
|     Self-Attentive Sequential Recommendation\|**SASRec**     |  ICDM, 2018  |      UCSD      |
| Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding\|**Caser** |  WSDM, 2018  |   Jiaxi Tang   |
| Next Item Recommendation with Self-Attentive Metric Learning\|**AttRec** | AAAAI, 2019  |  Shuai Zhang   |
| FISSA: Fusing Item Similarity Models with Self-Attention Networks for Sequential Recommendation\|**FISSA** | RecSys, 2020 |    Jing Lin    |

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

1. If you have any suggestions or questions about the project, you can leave a comment on `Issue`.
2. wechat：

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/weixin.jpg" width="20%"/></div>

