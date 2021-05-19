## AFM

### 1. 论文
Attentional Factorization Machines: Learning the Weight of Feature Interactions via Attention Networks

**创新**：基于Attention的Pooling层，与一般的Attention机制不同，具体可以看原文笔记。  

原文笔记：  https://mp.weixin.qq.com/s/hPCS9Dw2vT2pwdWwPo0EJg



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_9.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用Criteo数据集进行测试。数据集的处理见`../data_process`文件，主要分为：

1. 考虑到Criteo文件过大，因此可以通过`read_part`和`sample_sum`读取部分数据进行测试；
2. 对缺失数据进行填充；
3. 对密集数据`I1-I13`进行离散化分桶（bins=100），对稀疏数据`C1-C26`进行重新编码`LabelEncoder`；
4. 整理得到`feature_columns`；
5. 切分数据集，最后返回`feature_columns, (train_X, train_y), (test_X, test_y)`；



### 4. 模型API

```python
class AFM(Model):
    def __init__(self, feature_columns, mode, att_vector=8, activation='relu', dropout=0.5, embed_reg=1e-6):
        """
        AFM 
        :param feature_columns: A list. sparse column feature information.
        :param mode: A string. 'max'(MAX Pooling) or 'avg'(Average Pooling) or 'att'(Attention)
        :param att_vector: A scalar. attention vector.
        :param activation: A string. Activation function of attention.
        :param dropout: A scalar. Dropout.
        :param embed_reg: A scalar. the regularizer of embedding
        """
```



### 5. 实验超参数

- file：Criteo文件；
- read_part：是否读取部分数据，`True`；
- sample_num：读取部分时，样本数量，`5000000`；
- test_size：测试集比例，`0.2`；
- 
- embed_dim：Embedding维度，`8`；
- att_vector：attention层隐藏单元，`8`；
- mode：Pooling的类型, `att`；
- dropout：`0.5`;
- activation：`relu`；
- embed_reg：`1e-5`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`4096`；
- epoch：`10`；



### 6. 实验结果

1. 采用Criteo数据集中前`500w`条数据，最终测试集的结果为：
   - max：`AUC：0.758616`；
   - avg：`AUC：0.716929`；【avg的效果特别差】
   - att：`AUC：0.753657`；
2. 采用Criteo数据集全部内容：
   - 学习参数：235,393,945；
   - 单个Epoch运行时间【GPU：Tesla V100S-PCI】：323s；
   - 测试集结果（att）： `AUC: 0.787504, loss: 0.4762`；

