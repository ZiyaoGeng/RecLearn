## AttRec

### 1. 论文
Next Item Recommendation with Self-Attentive Metric Learning

**创新**：**长短期用户兴趣表示**  



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_19.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

本次选择ml-1m中的`ratings.dat`：

- 正负样本划分：可以选择分数大于某个阈值为正样本，小于和未评分作为负样本；
- 构建训练集、验证集、测试集；
  - 训练集，选择`0～t-2`时间步的作为训练正样本，同时随机生成单个负样本，与用户信息构造三元组；
  - 验证集，选择`t-1`时间步作为验证集正样本，其他同上；
  - 测试集：选择`t`时间步作为测试集，选择100个随机负样本，与用户信息分别构造100个三元组形式；
- 生成用户特征列、物品特征列；



### 4. 模型API

```python
class AttRec(Model):
    def __init__(self, feature_columns, maxlen=40, mode='inner', gamma=0.5, w=0.5, embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param gamma: A scalar. if mode == 'dist', gamma is the margin.
        :param mode: A string. inner or dist.
        :param w: A scalar. The weight of short interest.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：阈值，`1`；
- maxlen：序列长度，`50`；
- 
- embed_dim：embedding维度，`32`；
- embed_reg：embedding正则化参数，`1e-6`；
- gamma：margin，`0.5`；
- mode：尝试采用不同的计算相似度的形式，`inner product`或欧式距离，`inner`；
- w：长短期兴趣权重，`0.5`；
- K：top@k，`10`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`20`；



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的结果为：采用内积的形式下，`hit_rate@K:0.74`（没有调参，dist的效果有点问题）

