## MF-BPR

### 1. 论文
BPR: Bayesian Personalized Ranking from Implicit Feedback

**创新**：**提出对损失函数**  



### 2. 实验数据集

本次选择ml-1m中的`ratings.dat`：

- 正负样本划分：可以选择分数大于某个阈值为正样本，小于和未评分作为负样本；
- 构建训练集、验证集、测试集；
  - 训练集，选择`0～t-2`时间步的作为训练正样本，同时随机生成单个负样本，与用户信息构造三元组；
  - 验证集，选择`t-1`时间步作为验证集正样本，其他同上；
  - 测试集：选择`t`时间步作为测试集，选择100个随机负样本，与用户信息分别构造100个三元组形式；
- 生成用户特征列、物品特征列；



### 4. 模型API

```python
class BPR(Model):
    def __init__(self, feature_columns, mode='inner', embed_reg=1e-6):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :mode: A string. 'inner' or 'dist'.
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：阈值，`1`；
- 
- embed_dim：embedding维度，`32`；
- mode：尝试采用不同的计算相似度的形式，`inner product`或`dist`欧式距离，`inner`；
- embed_reg：embedding正则化参数，`1e-6`；
- K：top@k，`10`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`20`；



### 6. 实验结果

采用测试集评估（1正样本，100负样本），结果：

- 采用inner：`hit_rate@K:0.5442`，与SASRec中的baseline差`0.03`；
- 采用dist：`hit_rate@K:0.4627`，与内积计算差距过大，需要调整；

