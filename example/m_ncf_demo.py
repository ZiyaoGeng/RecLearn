from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import NCF
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


epochs = 5
learning_rate = 0.001
batch_size = 512
neg_num = 2
k = 10


def main():
    zip_path = 'data/ml-1m.zip'
    fea_cols = {
        'user_num': 6040,
        'item_num': 4000,
        'embed_dim': 8
    }
    params = {
        'fea_cols': fea_cols,
        'hidden_units': [256, 128, 64],
        'activation': 'relu',
        'dnn_dropout': 0.3,
        'is_batch_norm': False,
        'loss_name': 'bpr_loss',
        'gamma': 0.3,
    }
    train_path, val_path, test_path = ml.load_zip_movielens(zip_path=zip_path)
    train_data = ml.load_ml(train_path, neg_num)
    val_data = ml.load_ml(val_path, neg_num)
    test_data = ml.load_ml(test_path, 100)

    model = NCF(**params)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    for epoch in range(1, epochs+1):
        model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=1,
            batch_size=batch_size
        )
        if epoch % 2 == 0 or epoch == epochs:
            eval_dict = eval_pos_neg(model, test_data, batch_size, ["hr", "ndcg"], k)
            print(eval_dict)


main()