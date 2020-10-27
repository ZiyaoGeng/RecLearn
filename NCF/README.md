## NCF

### 1. 论文
Neural network-based Collaborative Filtering

**创新**：**结合神经网络的协同过滤**  



### 2. 模型结构

<div align=center><img src="https://cdn.jsdelivr.net/gh/BlackSpaceGZY/cdn/img/tf_1.png" width="70%;" style="float:center"/></div>



### 3. 实验数据集

采用ml-1m数据集进行测试，将其处理为用户序列。数据集的处理见`utils`文件，该部分采用了原文github的处理方法；



### 4. 模型API

```python
class NeuMF(keras.Model):
    def __init__(self, num_users, num_items, mf_dim, layers, reg_layers, reg_mf):
        super(NeuMF, self).__init__()
```



### 5. 实验超参数

见`configs.py`文件。



### 6. 实验结果

采用ml-1m数据集数据，最终测试集的结果为：

