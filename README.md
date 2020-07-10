# Recommended-System with TensorFlow 2.0

推荐系统的实验；

环境：

Python 3.7；

Tensorflow 2.0；



### Neural network-based Collaborative Filtering（NCF）

[NCF](NCF)



### Deep Interest Network for Click-Through Rate Prediction(DIN)

[DIN](DIN)

数据集为论文中的[Amazon Dataset](http://jmcauley.ucsd.edu/data/amazon/)，下载并解压：

```python
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Electronics_5.json.gz
gzip -d reviews_Electronics_5.json.gz
wget -c http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/meta_Electronics.json.gz
gzip -d meta_Electronics.json.gz
```

其中`reviews_Electronics_5.json`为用户的行为数据，`meta_Electronics`为广告的元数据。



### Wide & Deep Learning for Recommender Systems

[Wide&Deep](Wide&Deep)

数据同上，数据处理方式参考DIN中的内容。