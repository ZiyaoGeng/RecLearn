## DNN

### 1. DNN
采用简单的MLP方法来处理用户行为序列。



### 2. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件，主要分为：

1. 转换为隐式数据集，定义分数大于某个阈值为正样本，此处默认为`2`；
2. 数据集按照用户、时间排序，方便后续划分样本；
3. 正负样本1:1，因此生成对应的负样本，并且产生用户历史行为序列，特别的，对于测试集，`y`采用`[user_id, 1]`或`[user_id, 0]`的方式，是为了对单个用户进行排序，已得到指标`Hit`与`NDCG`；
4. 创建得到新的训练集、验证集、测试集，格式为：`'hist', 'target_item', 'label'`；
5. 由于序列的长度各不相同，因此需要使用`tf.keras.preprocessing.sequence.pad_sequences`方法进行填充/切割，此外，**由于序列中只有一个特征`item_id`，经过填充/切割后，维度会缺失，所以需要进行增添维度**；
6. 最后返回`item_fea_col, (train_X, train_y), (val_X, val_y), (test_X, test_y)`；



### 4. 模型API

```python
class DNN(tf.keras.Model):
    def __init__(self, item_fea_col, maxlen=40, hidden_units=128, activation='relu', embed_reg=1e-6):
        """
        DNN model
        :param item_fea_col: A dict contains 'feat_name', 'feat_num' and 'embed_dim'.
        :param maxlen: A scalar. Number of length of sequence.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：阈值，`1`；
- maxlen：序列长度，`100`；
- 
- embed_dim：embedding维度，`64`；
- hidden_unit：`256`	；
- embed_reg：embedding正则化参数，`1e-6`；
- activation：`relu`；
- K：top@k，`10`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`30`；



### 6. 实验结果

采用测试集评估（1正样本，100负样本），结果：**HR = 0.6363, NDCG = 0.3670**；
