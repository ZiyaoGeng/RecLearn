## 公开数据集介绍

### 1. Movielens



### 2. Amazon



### 3. Criteo

数据即的特征如下所示：

- Label：标签，表示目标广告点击（1）或未点击（0）；
- I1-I13：13个数值特征，也称为计数特征；
- C1-C26：26个分类特征（稀疏特征），为了匿名化的目的，这些特性的值被散列到32位上；

由于数据量过大，可以采取**读入部分数据**：

```python
data_df = pd.read_csv('dataset/train.txt', sep='\t', iterator=True, header=None,
                          names=names)
    data_df = data_df.get_chunk(100000)
```

前五个数据如下所示：

```python
   label   I1   I2    I3  ...       C23       C24       C25       C26
0      0  1.0    1   5.0  ...  3a171ecb  c5c50484  e8b83407  9727dd16
1      0  2.0    0  44.0  ...  3a171ecb  43f13e8b  e8b83407  731c3655
2      0  2.0    0   1.0  ...  3a171ecb  3b183c5c       NaN       NaN
3      0  NaN  893   NaN  ...  3a171ecb  9117a34a       NaN       NaN
4      0  3.0   -1   NaN  ...  32c7478e  b34f3128       NaN       NaN

```

数据存在缺失值，可以通过不同的方式进行**填充或删除**，为简单起见，处理方式为：

```python
sparse_features = ['C' + str(i) for i in range(1, 27)]
dense_features = ['I' + str(i) for i in range(1, 14)]

data_df[sparse_features] = data_df[sparse_features].fillna('-1')
data_df[dense_features] = data_df[dense_features].fillna(0)
```

对稀疏数据进行**标签编码**，转化为二元稀疏特征：

```python
for feat in sparse_features:
  le = LabelEncoder()
  data_df[feat] = le.fit_transform(data_df[feat])
```

对数值型数据进行**标准化**：

```python
mms = MinMaxScaler(feature_range=(0, 1))
data_df[dense_features] = mms.fit_transform(data_df[dense_features])
```

由于模型的建立需要部分数据特征，如数值特征和稀疏特征的个数，每个稀疏特征的embedding维度，因此我们建立一个特征列，用来存储上述信息：

稀疏特征：

- feat：特征名；
- feat_num：特征不重复的总个数；
- embed_dim：embedding维度；

数值特征：

- feat：特征名；

```python
def sparseFeature(feat, feat_num, embed_dim=4):
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

def denseFeature(feat):
    return {'feat': feat}
  
feature_columns = [[denseFeature(feat) for feat in dense_features]] +
                      [[sparseFeature(feat, len(data_df[feat].unique()), embed_dim=4) for feat in sparse_features]]

```

划分数据集：

```python
train, test = train_test_split(data_df, test_size=0.2)
```