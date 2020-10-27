## MF

### 1. 论文
 Matrix Factorization Techniques for Recommender Systems

**创新**：**经典的矩阵分解模型**  



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_14.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用`ml-1m`数据集进行测试，转换为隐式数据集。数据集的处理见`utils`文件，主要分为：
1. 读取数据，列名`'UserId', 'MovieId', 'Rating', 'Timestamp'`；
2. 每个用户的平均打分作为一个特征`mean`；
3. 得到`feature_columns`：密集数据`mena`，稀疏数据为`item_id、user_id`；
4. 统计每个用户电影评分的总数，将用户的80%作为训练集（按时间排序），20%作为测试集；
5. 最后返回`feature_columns, (train_X, train_y), (test_X, test_y)`；



### 4. 模型API

```python
class MF_layer(Layer):
    def __init__(self, user_num, item_num, latent_dim, use_bias=False, user_reg=1e-4, item_reg=1e-4,
                 user_bias_reg=1e-4, item_bias_reg=1e-4):
        """
        MF Layer
        :param user_num: user length
        :param item_num: item length
        :param latent_dim: latent number
        :param use_bias: whether using bias or not
        :param user_reg: regularization of user
        :param item_reg: regularization of item
        :param user_bias_reg: regularization of user bias
        :param item_bias_reg: regularization of item bias
        """
```



### 5. 实验超参数

- file：Amazon Electronic文件；
- test_size：测试集占比，`0.2`；
- 
- latent_dim：隐藏单元维度，`32`；
- use_bias：是否加入偏置，`True`；
- 
- learning_rate：学习率，`0.001`；
- batch_size：`512`；
- epoch：`10`；



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的结果为：

