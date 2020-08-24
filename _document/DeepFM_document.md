## DeepFM代码文档

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

- 稀疏特征embed_dim根据实际情况设置；



### 3. model.py

FM部分：

```python
class FM(layers.Layer):
	"""
	Wide part
	"""
	def __init__(self, k):
		"""

		:param k: the dimension of the latent vector
		"""
		super(FM, self).__init__()
		self.k = k

	def build(self, input_shape):
		self.w0 = self.add_weight(name='w0', shape=(1,), initializer=tf.zeros_initializer())
		self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
								 regularizer=regularizers.l2(0.01))
		self.V = self.add_weight(name='V', shape=(self.k, input_shape[-1]),
								 regularizer=regularizers.l2(0.01))

	def call(self, inputs):
		# first order
		first_order = self.w0 + tf.matmul(inputs, self.w)
		# second order
		second_order = 0.5 * tf.reduce_sum(
			tf.pow(tf.matmul(inputs, tf.transpose(self.V)), 2) -
			tf.matmul(tf.pow(inputs, 2), tf.pow(tf.transpose(self.V), 2)), axis=1, keepdims=True)
		return first_order + second_order
```

1、参数k为隐向量的维度；

2、w0，w，V分别对应偏置、1阶权重、2阶隐向量；

3、最后输出为一个数值，并不是按照模型图那样得到多个FM单元；



Deep部分：

一个简单的MLP

```python
class MLP(layers.Layer):
	"""
	Deep part
	"""
	def __init__(self, hidden_units, dropout_deep):
		"""

		:param hidden_units: list of hidden layer units's numbers
		:param dropout_deep: dropout number
		"""
		super(MLP, self).__init__()
		self.dnn_network = [Dense(units=unit, activation='relu') for unit in hidden_units[1:]]
		self.dropout = Dropout(dropout_deep)

	def call(self, inputs):
		x = inputs
		for dnn in self.dnn_network:
			x = dnn(x)
		x = self.dropout(x)
		return x
```

1、hidden_units：为隐藏单元列表，dropout_deep：为dropout的取值，当然可以自行调整其位置或者设置多个dropout；



DeepFM部分：

```python
class DeepFM(keras.Model):
	def __init__(self, feature_columns, k, hidden_units=None, dropout_deep=0.5):
		super(DeepFM, self).__init__()
		if hidden_units is None:
			hidden_units = [200, 200, 200]
		self.dense_feature_columns, self.sparse_feature_columns = feature_columns
		# embedding layer
		self.embed_layers = {
			'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
										 output_dim=feat['embed_dim'], embeddings_initializer='random_uniform')
			for i, feat in enumerate(self.sparse_feature_columns)
		}
		self.fm = FM(k)
		self.mlp = MLP(hidden_units, dropout_deep)
		self.dense = Dense(1, activation=None)
		self.w1 = self.add_weight(name='wide_weight', shape=(1,))
		self.w2 = self.add_weight(name='deep_weight', shape=(1,))

	def call(self, inputs):
		dense_inputs, sparse_inputs = inputs
		stack = dense_inputs
		for i in range(sparse_inputs.shape[1]):
			embed_i = self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i])
			stack = tf.concat([stack, embed_i], axis=-1)
		wide_outputs = self.fm(stack)
		deep_outputs = self.mlp(stack)
		deep_outputs = self.dense(deep_outputs)
		outputs = tf.nn.sigmoid(tf.add(self.w1 * wide_outputs, self.w2 * deep_outputs))
		return outputs

```

重点：

- 模型的初始化参数有4个：特征列信息、FM的隐藏维度、隐藏层单元列表、dropout数值；
- embed_layers：表示所有稀疏特征的embedding矩阵字典，一一对应。embed_1~embed_26；
- fm：FM模型；
- mlp：Deep模型；
- dense：Deep模型最后的神经网络层；
- w1，w2：分别为FM模型输出与Deep模型输出的权重值；



### 4. train.py

```python
def main(embed_dim, learning_rate, epochs, batch_size, k, hidden_units, dropout_deep=0.5):
    """
    feature_columns is a list and contains two dict：
    - dense_features: {feat: dense_feature_name}
    - sparse_features: {feat: sparse_feature_name, feat_num: the number of this feature}
    train_X: [dense_train_X, sparse_train_X]
    test_X: [dense_test_X, sparse_test_X]
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset(embed_dim)

    # ============================Build Model==========================
    model = DeepFM(feature_columns, k, hidden_units, dropout_deep=dropout_deep)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/deepfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(check_path, save_weights_only=True,
    #                                                 verbose=1, period=5)
    # =========================Compile============================
    model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=learning_rate),
                  metrics=[AUC()])
    # ===========================Fit==============================
    model.fit(
        train_X,
        train_y,
        epochs=epochs,
        # callbacks=[checkpoint],
        batch_size=batch_size,
        validation_split=0.1
    )
    # ===========================Test==============================
    print('test AUC: %f' % model.evaluate(test_X, test_y)[1])
```

主函数：main()

- 包含参数：稀疏特征的维度（对之前的代码进行更改了，在此处直接修改embed_dim）、学习率、训练的轮数、batch_size、k、隐藏单元列表、dropout；
- 损失函数：二元交叉熵；
- 优化器：Adam；