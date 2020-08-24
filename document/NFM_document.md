## NFM代码文档

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

NFM：

```python
class NFM(keras.Model):
    def __init__(self, feature_columns, hidden_units, dropout_rate):
        super(NFM, self).__init__()
        self.dense_feature_columns, self.sparse_feature_columns = feature_columns
        self.embed_layers = {
            'embed_' + str(i): Embedding(input_dim=feat['feat_num'], input_length=1,
                                         output_dim=feat['embed_dim'], embeddings_initializer='random_uniform')
            for i, feat in enumerate(self.sparse_feature_columns)
        }
        self.dropout = Dropout(rate=dropout_rate)
        self.bn = BatchNormalization()
        self.concat = Concatenate(axis=-1)
        self.dnn_network = [Dense(units=unit, activation='relu') for unit in hidden_units]
        self.dense = Dense(1)

    def call(self, inputs):
        # Inputs layer
        dense_inputs, sparse_inputs = inputs
        # Embedding layer
        embed = [self.embed_layers['embed_{}'.format(i)](sparse_inputs[:, i]) for i in range(sparse_inputs.shape[1])]
        embed = tf.transpose(tf.convert_to_tensor(embed), [1, 0, 2])
        # Bi-Interaction Layer
        embed = 0.5 * (tf.pow(tf.reduce_sum(embed, axis=1), 2) -
                       tf.reduce_sum(tf.pow(embed, 2), axis=1))
        # Concat
        x = self.concat([dense_inputs, embed])
        # Dropout
        x = self.dropout(x)
        # BatchNormalization
        x = self.bn(x)
        # Hidden Layers
        for dnn in self.dnn_network:
            x = dnn(x)
        outputs = tf.nn.sigmoid(self.dense(x))
        return outputs

```

重点：

- 模型的初始化参数有4个：特征列信息、隐藏层单元列表、dropout数值；

- embed_layers：表示所有稀疏特征的embedding矩阵字典，一一对应。embed_1~embed_26；

- 主要内容是按照公式的**Bi-Interaction Layer**：
  $$
  f_{B I}\left(\mathcal{V}_{x}\right)=\frac{1}{2}\left[\left(\sum_{i=1}^{n} x_{i} \mathbf{v}_{i}\right)^{2}-\sum_{i=1}^{n}\left(x_{i} \mathbf{v}_{i}\right)^{2}\right]
  $$
  



### 4. train.py

```python
"""
Created on August 2, 2020

train NFM model

@author: Ziyao Geng
"""

import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from utils import create_dataset
from model import NFM

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main(sample_num, embed_dim, learning_rate, epochs, batch_size, hidden_units, dropout_rate=0.5):
    """

    :param sample_num: the num of training sample
    :param embed_dim: the dimension of all embedding layer
    :param learning_rate:
    :param epochs:
    :param batch_size:
    :param hidden_units:
    :param dropout_rate:
    :return:
    """
    feature_columns, train_X, test_X, train_y, test_y = create_dataset(sample_num, embed_dim)

    # ============================Build Model==========================
    model = NFM(feature_columns, hidden_units, dropout_rate=dropout_rate)
    model.summary()
    # ============================model checkpoint======================
    # check_path = 'save/nfm_weights.epoch_{epoch:04d}.val_loss_{val_loss:.4f}.ckpt'
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


if __name__ == '__main__':
    sample_num = 100000
    embed_dim = 16
    learning_rate = 0.001
    batch_size = 512
    epochs = 5
    dropout_rate = 0.5
    # The number of hidden units in the deep network layer
    hidden_units = [256, 128, 64]
    main(sample_num, embed_dim, learning_rate, epochs, batch_size, hidden_units, dropout_rate)
```

主函数：main()

- 包含参数：稀疏特征的维度、学习率、训练的轮数、batch_size、k、隐藏单元列表、dropout；
- 损失函数：二元交叉熵；
- 优化器：Adam；