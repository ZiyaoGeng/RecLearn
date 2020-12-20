## SASRec

### 1. 论文
Self-Attentive Sequential Recommendation

**创新**：**结合自注意力的序列推荐**  

原文笔记：https://mp.weixin.qq.com/s/cRQi3FBi9OMdO7imK2Y4Ew



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_11.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件。



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
- test_neg_num：测试集物品数量，`100`；
- 
- embed_dim：Embedding维度，`50`；
- blocks：block的个数，`2`；
- num_heads：几头注意力，`1`；
- ffn_hidden_unit：FFN的隐藏单元，` 64`；
- dropout：`0.5`；
- norm_training：是否使用Layer Normalization，`True`；
- causality：是否使用，`False`；
- K：评价指标的@K，`10`；
- 
- learning_rate：学习率，`0.001`；
- epoch：`30`；
- batch_size：`512`；



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的评估结果为：**HR@10 = 0.8116, NDCG@10 = 0.5592**，HR与原文实验结果相差**0.0129**。

