## NCF

### 1. 论文
Neural network-based Collaborative Filtering

**创新**：**结合神经网络的协同过滤**  



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_1.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件。



### 4. 模型API

```python
class NCF(Model):
    def __init__(self, feature_columns, hidden_units=None, dropout=0.2, activation='relu', embed_reg=1e-6, **kwargs):
        """
        NCF model
        :param feature_columns: A list. user feature columns + item feature columns
        :param hidden_units: A list.
        :param dropout: A scalar.
        :param activation: A string.
        :param embed_reg: A scalar. The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：阈值，`1`；
- test_neg_num：测试集物品数量，`100`；
- 
- embed_dim：embedding维度，`32`；
- hidden_units：mlp的隐藏单元列表，`[256, 128, 64]`
- activation：`relu`；
- dropout：`0.2`；
- embed_reg：embedding正则化参数，`1e-6`；
- K：top@k，`10`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`20`；

### 6. 实验结果

采用ml-1m数据集数据（序列推荐的处理方法），最终测试集的结果为：**HR = 0.5265, NDCG = 0.2662**；【采用用户最后一个item进行预测，结果好像很差】

若使用论文的数据集，可通过`v1.0`版本进行测试；