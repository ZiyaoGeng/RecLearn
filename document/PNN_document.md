## PNN代码文档

> dataset：数据集
>
> >train.txt：Criteo数据集
>
> save：模型保存
>
> \_\_init\_\_.py
>
> model.py
>
> train.py
>
> utils.py



### 1. 数据集

数据集选用Criteo数据集。



### 2. utils.py

数据处理，主要分析见：[Criteo](../Dataset%20Introduction.md#3-criteo)。

- 数据集地址根据实际情况设置；

- `get_chunk`可设置训练的样本；

  ```python
  data_df.get_chunk(1000000)
  ```

- 稀疏特征embed_dim根据实际情况设置；**PNN的稀疏特征embedding维度必须都相同**



### 3. model.py

**模型构建的初始化函数：**

```python
class PNN(keras.Model):
    def __init__(self, feature_columns, embed_dim, hidden_units, mode='in'):
        super(PNN, self).__init__()
        # inner product or outer product
        self.mode = mode
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        # embedding dimension
        self.embed_dim = embed_dim
        # the number of feature fields
        self.field_num = len(self.sparse_feature_columns)
        # embedding layers
        # The embedding dimension of each feature field must be the same
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
                                         output_dim=self.embed_dim, embeddings_initializer='random_uniform')
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        # parameters
        self.w_z = self.add_weight(shape=(self.field_num, self.embed_dim, hidden_units[0]),
                                   initializer='random_uniform', name='w_z')
        if mode == 'in':
            self.w_p = self.add_weight(shape=(self.field_num, self.field_num,
                                              hidden_units[0]), initializer='random_uniform', name='w_p')
        else:
            self.w_p = self.add_weight(shape=(self.embed_dim, self.embed_dim,
                                              hidden_units[0]), initializer='random_uniform', name='w_p')
        self.l_b = self.add_weight(shape=(hidden_units[0], ), initializer='random_uniform', name='l_b')
        self.concat = Concatenate(axis=-1)
        # dnn
        self.dnn_network = [Dense(units=unit, activation='relu') for unit in hidden_units[1:]]
        self.dense_final = Dense(1)
```

1、模型的**初始化参数**：

- feature_columns：特征列信息；
- embed_dim；embedding维度，由于PNN每个稀疏特征域的维度都必须相同，因此我们直接在此处进行制定维度，忽略特征列中的信息；
- hidden_units：隐藏单元数；
- mode：特征交叉的方式：内积“in”或外积“out”；

2、模型的属性：

- field_num：所有稀疏特征域的数量；
- embed_layers：表示所有稀疏特征的embedding矩阵字典，一一对应。embed_1~embed_26；
- w_z：线性部分`z`进行全连接的权重参数，[field_num, embed_dim, hidden_unit]；
- w_p：平方部分`p`进行全连接的权重参数；
- l_b：拼接的偏置；
- dnn_network；



**call函数：**

```python
    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        z = embed
        if self.mode == 'in':
            p = tf.matmul(embed, tf.transpose(embed, [0, 2, 1]))
        else:
            f_sum = tf.reduce_sum(embed, axis=1, keepdims=True)
            p = tf.matmul(tf.transpose(f_sum, [0, 2, 1]), f_sum)

        l_z = tf.tensordot(z, self.w_z, axes=2)
        l_p = tf.tensordot(p, self.w_p, axes=2)
        l_1 = tf.nn.relu(self.concat([l_z + l_p + self.l_b, dense_inputs]))
        dnn_x = l_1
        for dense in self.dnn_network:
            dnn_x = dense(dnn_x)
        outputs = tf.nn.sigmoid(self.dense_final(dnn_x))
        return outputs
```

1. 得到数值特征和稀疏特征；
2. 得到当前物品对应的embedding向量列表；
3. 将向量列表转化张量，再将其按照[1, 0, 2]的顺序进行转置。**原因：**向量列表中的维度为[field_num, None, embed_dim]，`field_num`为所有稀疏特征域的数量，转置为：[None, field_num, embed_dim]
4. `z`即论文中的线性信号，与常数1作变换，并不改变内容；
5. 对于乘积操作分内外积，结果`p`是论文提到的平方信号：
   - 内积“in”：embed中的各个embedding向量进行内积操作，即embed矩阵与embed矩阵的转置进行内积运算（忽略None），即[None, field_num, embed_dim) * (None,  embed_dim, field_num) =[None, field_num, field_num]；--->与原文对应
   - 外积“out”：根据作者提出的**降维方法**，把所有两两特征Embedding向量外积互操作的结果叠加，即`f_sum`；故p的计算为[None, embed_dim, 1] * [None, 1, embed_dim] = [None, embed_dim, embed_dim]；---->与原文对应
6. 文章没有直接将上述的`z`与`p`直接拼接，而是做了一个局部的全连接。这里使用**tensordot**，具体可以看Tensorflow的官方文档。
   - `l_z`：z * w_z [None, field_num, embed_dim] * [field_num, embed_dim, hidden_unit]，后两个维度与前两个维度作点积，=[None,  hidden_unit]， `hidden_unit`指隐藏单元列表的第一个值；
   - `l_p`：p*l_p [None, embed_dim, embed_dim] * [embed_dim, embed_dim, hidden_unit] = [None, hidden_unit]
   - `l_1`：l_z、l_p、l_p、dense_input进行拼接，并经过relu激活函数；
7. 经过一个MLP，然后最后结果通过sigmoid激活函数输出。





### 4. train.py

```python
def main(learning_rate, epochs, embed_dim, hidden_units, mode='in'):
    """
    feature_columns is a list and contains two dict：
    - dense_features: {feat: dense_feature_name}
    - sparse_features: {feat: sparse_feature_name, feat_num: the number of this feature,
    embed_dim: the embedding dimension of this feature }
    train_X: [dense_train_X, sparse_train_X]
    test_X: [dense_test_X, sparse_test_X]
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset()

    # ============================Build Model==========================
    model = PNN(feature_columns, embed_dim, hidden_units)
    model.summary()
    # ============================model checkpoint======================
    check_path = 'save/pnn_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
                                                    verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        callbacks=[checkpoint],
        batch_size=128,
        validation_split=0.2
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])
```

主函数：main()

- 包含参数：学习率、训练的轮数、embedding维度、隐藏单元列表、乘积方式；
- 损失函数：二元交叉熵；
- 优化器：Adam；