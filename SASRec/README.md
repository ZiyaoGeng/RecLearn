## SASRec

### 1. 论文
Self-Attentive Sequential Recommendation

**创新**：**结合自注意力的序列推荐**  

原文笔记：https://mp.weixin.qq.com/s/cRQi3FBi9OMdO7imK2Y4Ew



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_11.png" width="70%;" style="float:center"/></div>



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
class SASRec(tf.keras.Model):
    def __init__(self, item_feat_col, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=40, norm_training=True, causality=False, embed_reg=1e-6):
        """
        SASRec model
        :param item_feat_col: A dict contains 'feat_name', 'feat_num' and 'embed_dim'.
        :param blocks: A scalar. The Number of blocks.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param maxlen: A scalar. Number of length of sequence
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        :param embed_reg: A scalar. The regularizer of embedding
        """
```



### 5. 实验超参数

- file：Amazon Electronic文件；
- trans_score：ml-1m分数转换，`1`；
- maxlen：最大序列长度，`200`；
- 
- embed_dim：Embedding维度，`32`；
- blocks：block的个数，`2`；
- num_heads：几头注意力，`1`；
- ffn_hidden_unit：FFN的隐藏单元，` 50`；
- dropout：`0.5`；
- norm_training：是否使用Layer Normalization，`True`；
- causality：是否使用，`False`；
- K：评价指标的@K，`10`；
- 
- learning_rate：学习率，`0.001`；
- epoch：`30`；
- batch_size：`512`；



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的结果为：`hit_rate@K：0.774`，与原文的`0.824`差了`0.05`；

