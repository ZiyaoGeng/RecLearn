## 公开数据集介绍

### 1. Movielens

ml-1m数据集中包含`ratings.dat`、`users.dat`、`movies.dat`。

**ratings.dat**

标签列表为：UserID::MovieID::Rating::Timestamp

- UserIDs：用户ID（1～6040）
- MovieIDs：电影ID（1～3952）
- Ratings：评分（1～5）
- Timestamp：时间戳

每个用户至少有20个评分记录（经过筛选）

**user.dat**

标签列表为： UserID::Gender::Age::Occupation::Zip-code

 - Gender：性别， "M"代表男， "F"代表女；
- Age：年龄，分为多个区间
        *  1:  "Under 18"
        * 18:  "18-24"
        * 25:  "25-34"
        * 35:  "35-44"
        * 45:  "45-49"
        * 50:  "50-55"
        * 56:  "56+"
- 职业：
        *  0:  "other" or not specified
        *  1:  "academic/educator"
        *  2:  "artist"
        *  3:  "clerical/admin"
        *  4:  "college/grad student"
        *  5:  "customer service"
        *  6:  "doctor/health care"
        *  7:  "executive/managerial"
        *  8:  "farmer"
        *  9:  "homemaker"
        * 10:  "K-12 student"
        * 11:  "lawyer"
        * 12:  "programmer"
        * 13:  "retired"
        * 14:  "sales/marketing"
        * 15:  "scientist"
        * 16:  "self-employed"
        * 17:  "technician/engineer"
        * 18:  "tradesman/craftsman"
        * 19:  "unemployed"
    
    * 20:  "writer"

**movies.dat**

标签列表为：MovieID::Title::Genres

- Titles：电影名称；
- Genres：电影分类
        * Action
        * Adventure
        * Animation
        * Children's
        * Comedy
        * Crime
        * Documentary
        * Drama
        * Fantasy
        * Film-Noir
        * Horror
        * Musical
        * Mystery
        * Romance
        * Sci-Fi
        * Thriller
        * War
        * Western

用于推荐系统或CTR预估，目前大多数是采用隐式反馈的形式，即将各个打分依据某个原则改变为1（高于某个分数）和0（低于某个分数或未打分），并且还要划分正负样本等。

以处理好的数据集：参考[NCF开源代码](https://github.com/hexiangnan/neural_collaborative_filtering)



### 2. Amazon

Amazon-Electronics数据集分为两部分：`reviews_Electronics_5.json`为用户的行为数据，`meta_Electronics`为广告的元数据。

`reviews`某单个样本如下：

```
{
  "reviewerID": "A2SUAM1J3GNN3B",
  "asin": "0000013714",
  "reviewerName": "J. McDonald",
  "helpful": [2, 3],
  "reviewText": "I bought this for my husband who plays the piano.  He is having a wonderful time playing these old hymns.  The music  is at times hard to read because we think the book was published for singing from more than playing from.  Great purchase though!",
  "overall": 5.0,
  "summary": "Heavenly Highway Hymns",
  "unixReviewTime": 1252800000,
  "reviewTime": "09 13, 2009"
}
```

各字段分别为：

- `reviewerID`：用户ID；
- `asin`： 物品ID；
- `reviewerName`：用户姓名；
- `helpful` ：评论帮助程度，例如上述`[2, 3]`表示为为`2/3`；
- `reviewText` ：文本信息；
- `overall` ：物品评分；
- `summary`：评论总结
- `unixReviewTime` ：时间戳
- `reviewTime` ：时间

`meta`某样本如下：

```python
{
  "asin": "0000031852",
  "title": "Girls Ballet Tutu Zebra Hot Pink",
  "price": 3.17,
  "imUrl": "http://ecx.images-amazon.com/images/I/51fAmVkTbyL._SY300_.jpg",
  "related":
  {
    "also_bought": ["B00JHONN1S", "B002BZX8Z6", "B00D2K1M3O", "0000031909", ..., "B00E1YRI4C", "B008UBQZKU", "B00D103F8U", "B007R2RM8W"],
    "also_viewed": ["B002BZX8Z6", "B00JHONN1S", "B008F0SU0Y", "B00D23MC6W", ..., "B00BFXLZ8M"],
    "bought_together": ["B002BZX8Z6"]
  },
  "salesRank": {"Toys & Games": 211836},
  "brand": "Coxlures",
  "categories": [["Sports & Outdoors", "Other Sports", "Dance"]]
}
```

各字段分别为：

- `asin` ：物品ID；
- `title` ：物品名称；
- `price` ：物品价格；
- `imUrl` ：物品图片的URL；
- `related` ：相关产品(也买，也看，一起买，看后再买)；
- `salesRank`： 销售排名信息；
- `brand` ：品牌名称；
- `categories` ：该物品属于的种类列表；

数据集的具体处理方法见知乎文章：[2018阿里CTR预估模型---DIN（深度兴趣网络），后附TF2.0复现代码](https://zhuanlan.zhihu.com/p/145149051)



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