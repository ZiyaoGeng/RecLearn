## xDeepFM

### 1. 论文
xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems

**创新**：CIN层是关键！！！  

原文笔记： https://mp.weixin.qq.com/s/TohOmVpQzNlA3vXv0gpobg

### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_12.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用Criteo数据集进行测试。数据集的处理见`../data_process`文件，主要分为：

1. 考虑到Criteo文件过大，因此可以通过`read_part`和`sample_sum`读取部分数据进行测试；
2. 对缺失数据进行填充；
3. 对密集数据`I1-I13`进行离散化分桶（bins=100），对稀疏数据`C1-C26`进行重新编码`LabelEncoder`；
4. 整理得到`feature_columns`；
5. 切分数据集，最后返回`feature_columns, (train_X, train_y), (test_X, test_y)`；



### 4. 模型API

```python
class xDeepFM(Model):
    def __init__(self, feature_columns, hidden_units, cin_size, dnn_dropout=0, dnn_activation='relu',
                 embed_reg=1e-6, cin_reg=1e-6, w_reg=1e-6):
        """
        xDeepFM
        :param feature_columns: A list. sparse column feature information.
        :param hidden_units: A list. a list of dnn hidden units.
        :param cin_size: A list. a list of the number of CIN layers.
        :param dnn_dropout: A scalar. dropout of dnn.
        :param dnn_activation: A string. activation function of dnn.
        :param embed_reg: A scalar. The regularizer of embedding.
        :param cin_reg: A scalar. The regularizer of cin.
        :param w_reg: A scalar. The regularizer of Linear.
        """
```



### 5. 实验超参数

- file：Criteo文件；
- read_part：是否读取部分数据，`True`；
- sample_num：读取部分时，样本数量，`5000000`；
- test_size：测试集比例，`0.2`；
- 
- embed_dim：Embedding维度，`8`；
- dnn_dropout：Dropout, `0.5`；
- hidden_unit：DNN的隐藏单元，`[256, 128, 64]`；
- cin_size：CIN尺度，`(128, 128)`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`4096`；
- epoch：`10`；



### 6. 实验结果

1. 采用Criteo数据集中前`500w`条数据，最终测试集的结果为：`AUC: 0.783963, loss: 0.4690`；
2. 采用Criteo数据集全部内容：
   - 学习参数：265,457,049；
   - 单个Epoch运行时间【GPU：Tesla V100S-PCI】：596s；
   - 测试集结果：`AUC: 0.791982, loss: 0.4696`；

