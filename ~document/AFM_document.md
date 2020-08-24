## AFM 代码文档

### 1. 数据集

数据集选用Criteo数据集。



### 2. utils.py

数据处理，主要分析见：[Criteo](../Dataset%20Introduction.md#3-criteo)。

- 数据集地址根据实际情况设置；

- `get_chunk`可设置训练的样本；

  ```python
  data_df.get_chunk(1000000)
  ```

- 稀疏特征embed_dim根据实际情况设置；



### 3. model.py

AFM建模部分最主要的是两部分：

（1）特征交互层，每个embedding向量与其余向量对应元素相乘（element-wise-product）

```python
# Pair-wise Interaction Layer
        element_wise_product_list = []
        for i in range(embed.shape[1]):
            for j in range(i+1, embed.shape[1]):
                element_wise_product_list.append(tf.multiply(embed[:, i], embed[:, j]))
        element_wise_product = tf.transpose(tf.convert_to_tensor(element_wise_product_list), [1, 0, 2])
```

（2）Attention部分；实验想与使用pooling的两种方式进行对比，因此加入mode参数："max"，"avg"，"att"：

```python
if self.mode == 'max':
  # MaxPooling Layer
  x = self.max(element_wise_product)
elif self.mode == 'avg':
  # AvgPooling Layer
  x = self.avg(element_wise_product)
else:
  # Attention Layer
  x = self.attention(element_wise_product)
```

关于本篇文章最重要的内容即是Attention部分：

```python'
    def attention(self, keys):
        a = self.attention_dense(keys)
        a = self.attention_dense2(a)
        a_score = tf.nn.softmax(a)
        a_score = tf.transpose(a_score, [0, 2, 1])
        outputs = tf.reshape(tf.matmul(a_score, keys), shape=(-1, keys.shape[2]))
        return outputs
```

经过两层全连接，以及softmax，得到对应的各个embedding向量的评分；再通过加权求和，得到最终的向量。





### 4. train.py

加入attention_unit，对应论文中的$k$