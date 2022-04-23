# 数据划分

在算法模型的实验中，首先需要完成所需数据集的划分。`reclearn`在`reclearn/data/datasets`中提供了ml-1m、beauty、games、steam、criteo数据集的划分方法。当然读者也可以自己设计方案，本文主要对基本的方法进行简要说明。

数据划分主要完成以下任务：

1. user、item索引的重新编码；
2. 训练集、验证集、测试集的划分，分别进行存储；
3. 记录user、item重新编码后的最大数值，为了指定embedding table的大小；

ml-1m、beauty、games、steam数据集划分方式相差不大，统一进行说明。而Criteo由于其庞大的数据量，需要单独说明。



## Moivelens等

上述我们已经阐明了主要的任务，但在这基础上还需要注意以下几点：

- 编码需要从1开始，0用于序列的填充。
- 取每个用户最后一次的交互行为作为测试集，最后第二次的行为作为验证集，其他作为训练集。

因此，例如`data/datasets/movielens.py`中的`split_data`方法：

```python
def split_data(file_path):
    """split movielens for general recommendation
        Args:
            :param file_path: A string. The file path of 'ratings.dat'.
        :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "ml_train.txt")
    val_path = os.path.join(dst_path, "ml_val.txt")
    test_path = os.path.join(dst_path, "ml_test.txt")
    meta_path = os.path.join(dst_path, "ml_meta.txt")
    users, items = set(), set()
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split("::")
            users.add(int(user))
            items.add(int(item))
            history.setdefault(int(user), [])
            history[int(user)].append([item, timestamp])
    random.shuffle(list(users))
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist = history[int(user)]
            hist.sort(key=lambda x: x[1])
            for idx, value in enumerate(hist):
                if idx == len(hist) - 1:
                    f3.write(str(user) + '\t' + value[0] + '\n')
                elif idx == len(hist) - 2:
                    f2.write(str(user) + '\t' + value[0] + '\n')
                else:
                    f1.write(str(user) + '\t' + value[0] + '\n')
    with open(meta_path, 'w') as f:
        f.write(str(max(users)) + '\t' + str(max(items)))
    return train_path, val_path, test_path, meta_path
```

由于ml-1m数据集user、item的编码都是从1开始，因此无需重新编码。

当然在reclearn项目中大部分模型需要使用**用户的行为序列作为特征**，因此还提供了序列数据的划分方式。例如`data/datasets/games.py`中的`split_seq_data`方法：

```python
def split_seq_data(file_path):
    """split amazon games for sequence recommendation
    Args:
        :param file_path: A string. The file path of 'ratings_Beauty.dat'.
    :return: train_path, val_path, test_path, meta_path
    """
    dst_path = os.path.dirname(file_path)
    train_path = os.path.join(dst_path, "games_seq_train.txt")
    val_path = os.path.join(dst_path, "games_seq_val.txt")
    test_path = os.path.join(dst_path, "games_seq_test.txt")
    meta_path = os.path.join(dst_path, "games_seq_meta.txt")
    users, items = set(), dict()
    user_idx, item_idx = 1, 1
    history = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            user, item, score, timestamp = line.strip().split(",")
            users.add(user)
            if items.get(item) is None:
                items[item] = str(item_idx)
                item_idx += 1
            history.setdefault(user, [])
            history[user].append([items[item], timestamp])
    with open(train_path, 'w') as f1, open(val_path, 'w') as f2, open(test_path, 'w') as f3:
        for user in users:
            hist_u = history[user]
            if len(hist_u) < 4:
                continue
            hist_u.sort(key=lambda x: x[1])
            hist = [x[0] for x in hist_u]
            time = [x[1] for x in hist_u]
            f1.write(str(user_idx) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + '\n')
            f2.write(str(user_idx) + "\t" + ' '.join(hist[:-2]) + "\t" + ' '.join(time[:-2]) + "\t" + hist[-2] + '\n')
            f3.write(str(user_idx) + "\t" + ' '.join(hist[:-1]) + "\t" + ' '.join(time[:-1]) + "\t" + hist[-1] + '\n')
            user_idx += 1
    with open(meta_path, 'w') as f:
        f.write(str(user_idx - 1) + '\t' + str(item_idx - 1))
    return train_path, val_path, test_path, meta_path
```



## Criteo

Criteo数据集大概有4500w的数据量，我们很难向处理moivelens那样将所有的数据都读如内存中。本文给出了两种处理方式：

1. 只读取部分数据进行实验；
2. 将数据集进行切分若干份，模型训练时分别读取，这样就避免了难以读入的问题；

**第一种方法**在`data/datasets/criteo.py`的`create_small_criteo_dataset`方法中：

```python
def create_small_criteo_dataset(file, embed_dim=8, read_part=True, sample_num=100000, test_size=0.2):
    """Load small criteo data(sample num) without splitting "train.txt".
    Note: If you want to load all data in the memory, please set "read_part" to False.
    Args:
        :param file: A string. dataset's path.
        :param embed_dim: A scalar. the embedding dimension of sparse features.
        :param read_part: A boolean. whether to read part of it.
        :param sample_num: A scalar. the number of instances if read_part is True.
        :param test_size: A scalar(float). ratio of test dataset.
    :return: feature columns such as [sparseFeature1, sparseFeature2, ...],
             train, such as  ({'C1': [...], 'C2': [...]]}, [1, 0, 1, ...])
             and test ({'C1': [...], 'C2': [...]]}, [1, 0, 1, ...]).
    """
    if read_part:
        data_df = pd.read_csv(file, sep='\t', iterator=True, header=None,
                          names=NAMES)
        data_df = data_df.get_chunk(sample_num)
    else:
        data_df = pd.read_csv(file, sep='\t', header=None, names=NAMES)
```

通过`read_part`参数判断是否需要读取部分数据，`sample_num`为读取数据的总量。



**第二种方法**具体需要考虑的内容：

1. 指定每个子数据集的数据量；
2. 子数据集存储的位置以及命名方式；
3. 如何实现数据集分割；

通过`data/datasets/criteo.py`的`get_split_file_path`方法可以完成数据集的分割以及最终子数据集的相对存储路径获取：

```python
def get_split_file_path(parent_path=None, dataset_path=None, sample_num=5000000):
    """Get the list of split file path.
    Note: Either parent_path or dataset_path must be valid.
    If exists dataset_path + "/split", parent_path = dataset_path + "/split".
    Args:
        :param parent_path: A string. split file's parent path.
        :param dataset_path: A string.
        :param sample_num: A int. The sample number of every split file.
    :return: A list. [file1_path, file2_path, ...]
    """
    sub_dir_name = 'split'
    if parent_path is None and dataset_path is None:
        raise ValueError('Please give parent path or file path.')
    if parent_path is None and os.path.exists(os.path.join(os.path.dirname(dataset_path), sub_dir_name)):
        parent_path = os.path.join(os.path.dirname(dataset_path), sub_dir_name)
    elif parent_path is None or not os.path.exists(parent_path):
        splitByLineCount(dataset_path, sample_num, sub_dir_name)
        parent_path = os.path.join(os.path.dirname(dataset_path), sub_dir_name)
    split_file_name = os.listdir(parent_path)
    split_file_name.sort()
    split_file_list = [parent_path + "/" + file_name for file_name in split_file_name if file_name[-3:] == 'txt']
    return split_file_list
```

`parent_path`为存储子数据集的路径，`dataset_path`为原始数据集的路径，这里也判断了子数据集是否已经分割好，若之前已经完成，则可以直接获取子数据集的列表路径，避免重复的完成任务。若没有完成，我没通过`data/utils.py`的`splitByLineCount`方法完成：

```python
def splitByLineCount(filename, count, sub_dir_name):
    """Split File.
    Note: You can specify how many rows of data each sub file contains.
    Args:
        :param filename: A string.
        :param count: A scalar(int).
        :param sub_dir_name: A string.
    :return:
    """
    f = open(filename, 'r')
    try:
        head = f.readline()
        buf = []
        sub = 1
        for line in f:
            buf.append(line)
            if len(buf) == count:
                sub = mkSubFile(buf, head, filename, sub_dir_name, sub)
                buf = []
        if len(buf) != 0:
            mkSubFile(buf, head, filename, sub_dir_name, sub)
    finally:
        f.close()
```

其中`mkSubFile`方法对当前达标目标说的内容`buf`进行存储，存储名字通过`sub`来指定。

```python
def mkSubFile(lines, head, srcName, sub_dir_name, sub):
    """Write sub-data.
    Args:
        :param lines: A list. Several pieces of data.
        :param head: A string. ['label', 'I1', 'I2', ...].
        :param srcName: A string. The name of data.
        :param sub_dir_name: A string.
        :param sub: A scalar(Int). Record the current number of sub file.
    :return: sub + 1.
    """
    root_path, file = os.path.split(srcName)
    file_name, suffix = file.split('.')
    split_file_name = file_name + "_" + str(sub).zfill(2) + "." + suffix
    split_file = os.path.join(root_path, sub_dir_name, split_file_name)
    if not os.path.exists(os.path.join(root_path, sub_dir_name)):
        os.mkdir(os.path.join(root_path, sub_dir_name))
    print('make file: %s' % split_file)
    f = open(split_file, 'w')
    try:
        f.writelines([head])
        f.writelines(lines)
        return sub + 1
    finally:
        f.close()
```

以上就完成了大数据集的切分，我们可以挑选任意一个子数据集作为测试集。