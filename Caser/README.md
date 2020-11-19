## Caser

### 1. 论文
Personalized Top-N Sequential Recommendation via Convolutional Sequence Embedding

**创新**：**采用卷积神经网络来提取用户的短期偏好**  

原文笔记：https://mp.weixin.qq.com/s/N_CDMlDDq1YMEqAp4Xw_dg



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_18.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件，主要分为：
1. 转换为隐式数据集，定义分数大于某个阈值为正样本，此处默认为`2`；
2. 数据集按照用户、时间排序，方便后续划分样本；
3. 正负样本1:1，因此生成对应的负样本，并且产生用户历史行为序列，特别的，对于测试集，`y`采用`[user_id, 1]`或`[user_id, 0]`的方式，是为了对单个用户进行排序，已得到指标`Hit`与`NDCG`；
4. 得到`feature_columns`：无密集数据，稀疏数据为`item_id`；
5. 生成用户行为列表，方便后续序列Embedding的提取，在此处，即`item_id`；
6. 打乱三个数据集；
7. 创建得到新的训练集、验证集、测试集，格式为：`'hist', 'target_item', 'label'`；
8. 由于序列的长度各不相同，因此需要使用`tf.keras.preprocessing.sequence.pad_sequences`方法进行填充/切割，此外，**由于序列中只有一个特征`item_id`，经过填充/切割后，维度会缺失，所以需要进行增添维度**；
9. 最后返回`feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)`；



### 4. 模型API

```python
class Caser(Model):
    def __init__(self, feature_columns, maxlen=40, hor_n=2, hor_h=8, ver_n=8, activation='relu', embed_reg=1e-6):
        """
        AttRec
        :param feature_columns: A feature columns list. user + seq
        :param maxlen: A scalar. In the paper, maxlen is L, the number of latest items.
        :param hor_n: A scalar. The number of horizontal filters.
        :param hor_h: A scalar. Height of horizontal filters.
        :param ver_n: A scalar. The number of vertical filters.
        :param activation: A string. 'relu', 'sigmoid' or 'tanh'.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：ml-1m分数转换，`1`；
- maxlen：最大序列长度，`200`；
- 
- embed_dim：Embedding维度，`32`；
- hor_n：水平过滤器的个数，`8`；
- hor_h：水平过滤器的高，`2`；
- ver_n：垂直过滤器的个数，`8`；
- dropout：`0.5`；
- activation：`relu`；
- K：评价指标的@K，`10`；
- 
- learning_rate：学习率，`0.001`；
- epoch：`30`；
- batch_size：`512`；



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的结果为：

