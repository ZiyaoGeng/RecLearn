## Caser

### 1. 论文
Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding

**创新**：**采用卷积神经网络来提取用户的短期偏好**  

原文笔记：https://mp.weixin.qq.com/s/N_CDMlDDq1YMEqAp4Xw_dg



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_20.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件，主要分为：
1. 转换为隐式数据集，定义分数大于某个阈值为正样本，此处默认为`1`；
2. 数据集按照用户、时间排序，方便后续划分样本；
3. 正负样本1:1，因此生成对应的负样本，并且产生用户历史行为序列；
4. 得到`feature_columns`；
7. 创建得到新的训练集、验证集、测试集；
8. 由于序列的长度各不相同，因此需要使用`tf.keras.preprocessing.sequence.pad_sequences`方法进行填充/切割；
9. 最后返回`feature_columns, (train_X, train_y), (val_X, val_y), (test_X, test_y)`；



### 4. 模型API

```python
class Caser(Model):
    def __init__(self, feature_columns, maxlen=40, hor_n=2, hor_h=8, ver_n=8, dropout=0.5, activation='relu', embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param hor_n: A scalar. The number of horizontal filters.
        :param hor_h: A scalar. Height of horizontal filters.
        :param ver_n: A scalar. The number of vertical filters.
        :param dropout: A scalar. The number of dropout.
        :param activation: A string. 'relu', 'sigmoid' or 'tanh'.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：ml-1m分数转换，`1`；
- maxlen：最大序列长度，`200`；
- 
- embed_dim：Embedding维度，`50`；
- hor_n：水平过滤器的个数，`8`；
- hor_h：水平过滤器的高，`2`；
- ver_n：垂直过滤器的个数，`4`；
- dropout：`0.2`；
- activation：`relu`；
- embed_reg：`1e-6`；
- K：评价指标的@K，`10`；
- 
- learning_rate：学习率，`0.001`；
- epoch：`30`；
- batch_size：`512`；



### 6. 实验结果

采用测试集评估（1正样本，100负样本），结果：**hit_rate@k: 0.7704【应该能更高】**，与SASRec中的Caser结果相差0.01，终于相差不大；

