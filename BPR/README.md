## MF-BPR

### 1. 论文
BPR: Bayesian Personalized Ranking from Implicit Feedback

**创新**：**提出对损失函数**  



### 2. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件。



### 4. 模型API

```python
class BPR(Model):
    def __init__(self, feature_columns, mode='inner', embed_reg=1e-6):
        """
        BPR
        :param feature_columns: A list. user feature columns + item feature columns
        :mode: A string. 'inner' or 'dist'.
        :param embed_reg: A scalar.  The regularizer of embedding.
        """
```



### 5. 实验超参数

- file：ml-1m文件；
- trans_score：阈值，`1`；
- test_neg_num：测试集物品数量，`100`；
- 
- embed_dim：embedding维度，`32`；
- mode：尝试采用不同的计算相似度的形式，`inner product`或`dist`欧式距离，`inner`；
- embed_reg：embedding正则化参数，`1e-6`；
- K：top@k，`10`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`20`；



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的评估结果为：

- 采用inner：**HR = 0.5349, NDCG = 0.2792**
- 采用dist：**HR = 0.4575, NDCG = 0.2311**

