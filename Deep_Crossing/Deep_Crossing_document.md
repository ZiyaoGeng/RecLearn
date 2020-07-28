## Deep Crossing代码文档

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

`get_chunk`可设置训练的样本

```
data_df.get_chunk(1000000)
```



### 3. model.py

**残差单元的构建：**

```python
class Residual_Units(Layer):
    """
    Residual Units
    """
    def __init__(self, hidden_unit, dim_stack):
        """
        :param hidden_unit: the dimension of cross layer unit
        :param dim_stack: the dimension of inputs unit
        """
        super(Residual_Units, self).__init__()
        self.layer1 = Dense(units=hidden_unit, activation='relu')
        self.layer2 = Dense(units=dim_stack)
        self.relu = ReLU()

    def call(self, inputs):
        x = inputs
        x = self.layer1(x)
        x = self.layer2(x)
        outputs = self.relu(x + inputs)
        return outputs
```

重点是**两层全连接层的隐藏单元**，为了使输出与输入的维度相同，最后一层的隐藏单元必须是输入的维度，即`dim_stack`。`hidden_unit`为第一层的隐藏单元。输入与输出相加后需要经过一个激活函数`Relu`。



**模型建模：**

```python
class Deep_Crossing(keras.Model):
    def __init__(self, feature_columns, hidden_units):
        super(Deep_Crossing, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
                                         output_dim=feat['embed_dim'], embeddings_initializer='random_uniform')
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        embed_dim = sum([feat['embed_dim'] for feat in self.sparse_feature_columns])
        dim_stack = len(self.dense_feature_columns) + embed_dim
        self.res_network = [Residual_Units(unit, dim_stack) for unit in hidden_units]
        self.dense = Dense(1)

    def call(self, inputs):
        dense_inputs, sparse_inputs = inputs
        stack = dense_inputs
        for i in range(sparse_inputs.shape[1]):
            embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
            stack = tf.concat([stack, embed_i], axis=-1)
        r = stack
        for res in self.res_network:
            r = res(r)
        outputs = tf.nn.sigmoid(self.dense(r))
        return outputs
```

重点：

- 模型的初始化参数有三个：feature_columns（特征列信息）、hidden_units（隐藏单元列表）。其中feature_columns具体信息见`utils.py`。hidden_units表示每层残差网络的隐藏单元数（不包括1）。

- embed_layers：表示所有稀疏特征的embedding矩阵字典，一一对应。embed_1~embed_26；

- embed_dim：所有的embedding输出的总维度；

- dim_stack：stack的维度，因为残差块需要输入维度，即数值特征个数+每个稀疏特征的embedding维度；

- res_network：残差网络（不带有卷积核）；

- 下面代码表示所有的向量进行拼接：

  ```python
  for i in range(sparse_inputs.shape[1]):
              embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
              stack = tf.concat([stack, embed_i], axis=-1)
  ```

- outputs：最后的结果输出



### 4. train.py

```python
def main(learning_rate, epochs, hidden_units):
    """
    feature_columns is a list and contains two dict：
    - dense_features: {feat: dense_feature_name}
    - sparse_features: {feat: sparse_feature_name, feat_num: the number of this feature}
    train_X: [dense_train_X, sparse_train_X]
    test_X: [dense_test_X, sparse_test_X]
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset()

    # ============================Build Model==========================
    model = Deep_Crossing(feature_columns, hidden_units)
    model.summary()
    # ============================model checkpoint======================
    check_path = 'save/deep_crossing_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
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

- 包含参数：学习率、训练的轮数、隐藏单元列表；
- 损失函数：二元交叉熵；
- 优化器：Adam；

