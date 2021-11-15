from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import BPR
from reclearn.data.datasets import movielens as ml

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def main():
    epochs = 5
    learning_rate = 0.001
    batch_size = 512
    neg_num = 2
    zip_path = 'data/ml-1m.zip'
    train_path, val_path, test_path = ml.load_zip_movielens(zip_path=zip_path)
    fea_cols = {
        'user_num': 6040,
        'item_num': 4000,
        'embed_dim': 8
    }
    # data = ml.load_ml(train_path, neg_num)
    dataset = ml.generate_ml(train_path, neg_num)
    dataset = dataset.map(lambda user, pos, neg: {'user': user, 'pos_item': pos, 'neg_item': neg}).shuffle(batch_size).batch(batch_size)
    model = BPR(fea_cols)
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    model.fit(
        x=dataset,
        epochs=epochs,
        # batch_size=batch_size
    )


main()