## xDeepFM

### 1. 论文
Deep Interest Network for Click-Through Rate Prediction

**创新**：兴趣的提取  

原文笔记： https://mp.weixin.qq.com/s/uIs_FpeowSEpP5fkVDq1Nw



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_2.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用Amazon数据集中的Electronic子集进行测试。数据集的处理见`preprocess`文件夹和`utils`文件，主要分为：
1. 采用`preprocess`中的`1_convert_pd.py`和`2_remap_id.py`文件得到`remap.pkl`处理后的数据（原文github采用该方法，此处不再更改）；
3. 读取数据，更改列名：`'user_id', 'item_id', 'time'`；
4. 正负样本1:1，因此生成对应的负样本，并且产生用户历史行为序列；
4. 得到`feature_columns`：无密集数据，稀疏数据为`item_id`和`cate_id`；
5. 生成用户行为列表，方便后续序列Embedding的提取，在此处，即`item_id, cate_id`；
6. 得到新的训练集、验证集、测试集，格式为：`'hist', 'target_item', 'label'`；
7. 由于序列的长度各不相同，因此需要使用`tf.keras.preprocessing.sequence.pad_sequences`方法进行填充；
8. 最后返回`feature_columns, behavior_list, (train_X, train_y), (val_X, val_y), (test_X, test_y)`；



### 4. 模型API

```python
class DIN(Model):
    def __init__(self, feature_columns, behavior_feature_list, att_hidden_units=(80, 40), 
        ffn_hidden_units=(80, 40), att_activation='sigmoid', ffn_activation='prelu', maxlen=40, dropout=0., embed_reg=1e-4):
        """
        DIN
        :param feature_columns: A list. dense_feature_columns + sparse_feature_columns
        :param behavior_feature_list: A list. the list of behavior feature names
        :param att_hidden_units: A tuple or list. Attention hidden units.
        :param ffn_hidden_units: A tuple or list. Hidden units list of FFN.
        :param att_activation: A String. The activation of attention.
        :param ffn_activation: A String. Prelu or Dice.
        :param maxlen: A scalar. Maximum sequence length.
        :param dropout: A scalar. The number of Dropout.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：Amazon Electronic文件；
- maxlen：最大序列长度，`40`；
- 
- embed_dim：Embedding维度，`32`；
- att_hidden_units：Attention中的全连接隐藏单元，`[80, 40]`；
- ffn_hidden_units：FFN中的全连接隐藏单元，`[256, 128, 64]`；
- dnn_dropout：Dropout， `0.5`；
- att_activation：Attention中全连接的激活函数，`sigmoid`；
- ffn_activation：FFN中的激活函数，Prelu或Dice，`prelu`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`5`；



### 6. 实验结果

采用Amazon-Electronic数据集数据，最终测试集的结果为：`AUC：0.738484`