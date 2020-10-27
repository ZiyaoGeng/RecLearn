## STAMP

### 1. 论文
STAMP: Short-Term Attention/Memory Priority Model for Session-based Recommendation

**创新**：**结合长期记忆（序列兴趣）和短期记忆（当前兴趣）**  

原文笔记：https://mp.weixin.qq.com/s/TXOSQAkwky1d27PciKjqtQ



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_13.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用`Diginetica`数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件，主要分为：
1. 读取数据（可以取部分数据进行测试）；
2. 过滤掉session长度为1的样本；
3. 过滤掉包含某物品（出现次数小于5）的样本；
4. 对特征`itemId`进行`LabelEncoder`，将其转化为`0, 1,...`范围；
5. 按照`evetdate、sessionId`排序；
6. 按照`eventdate`划分训练集、验证集、测试集；
7. 生成序列【无负样本】，生成新的数据，格式为`hist, label`，因此需要使用`tf.keras.preprocessing.sequence.pad_sequences`方法进行填充/切割，此外，**由于序列中只有一个特征`item_id`，经过填充/切割后，维度会缺失，所以需要进行增添维度**；
8. 生成一个物品池`item pooling`：物品池按序号排序；
9. 得到`feature_columns`：无密集数据，稀疏数据为`item_id`；
10. 生成用户行为列表，方便后续序列Embedding的提取，在此处，即`item_id`；
11. 最后返回`feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)`；



### 4. 模型API

```python
class STAMP(tf.keras.Model):
    def __init__(self, feature_columns, behavior_feature_list, item_pooling, maxlen=40, activation='tanh', embed_reg=1e-4):
        """
        STAMP
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param item_pooling: A Ndarray or Tensor, shape=(m, n),
        m is the number of items, and n is the number of behavior feature. The item pooling.
        :param activation: A String. The activation of FFN.
        :param maxlen: A scalar. Maximum sequence length.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：Amazon Electronic文件；
- maxlen：最大序列长度，`40`；
- 
- embed_dim：Embedding维度，`100`；
- K：评价指标的@K，`20`；
- 

- learning_rate：学习率，`0.0015；
- batch_size：`128`；
- epoch：`30`；



### 6. 实验结果

采用Diginetica数据集数据，最终测试集的结果为：

