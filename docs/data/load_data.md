## 数据读取

数据读取是数据pipeline的最后一步。我们需要指定哪些特征的读取、负样本的数量（Top-K推荐中）以及一些特征工程。



## Movielens等

对应于`split_data`方法，`data/datasets/movielens.py`中有`load_data`：

```python
def load_data(file_path, neg_num, max_item_num):
    """load movielens dataset.
    Args:
        :param file_path: A string. The file path.
        :param neg_num: A scalar(int). The negative num of one sample.
        :param max_item_num: A scalar(int). The max index of item.
    :return: A dict. data.
    """
    data = np.array(pd.read_csv(file_path, delimiter='\t'))
    np.random.shuffle(data)
    neg_items = []
    for i in tqdm(range(len(data))):
        neg_item = [random.randint(1, max_item_num) for _ in range(neg_num)]
        neg_items.append(neg_item)
    return {'user': data[:, 0].astype(int), 'pos_item': data[:, 1].astype(int), 'neg_item': np.array(neg_items)}
```

给定数据集路径（训练/验证/测试）、负样本数量以及最大的item索引值（为了控制负样本的范围）。该方法并没有包含一些特征工程，只包括了`user id`、`item id`两个特征。