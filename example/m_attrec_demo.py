from tensorflow.keras.optimizers import Adam
from time import time
from reclearn.models.matching import AttRec
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '6'


def main():
    epochs = 20
    learning_rate = 0.001
    batch_size = 512
    neg_num = 1
    seq_len = 5
    k = 10
    test_neg_num = 100
    file_path = 'data/ml-1m/ratings.dat'
    user_num, item_num, train_path, val_path, test_path = ml.load_seq_movielens(file_path=file_path)
    train_data = ml.load_seq_ml(train_path, "train", neg_num, seq_len, contain_user=True)
    val_data = ml.load_seq_ml(val_path, "val", neg_num, seq_len, contain_user=True)
    test_data = ml.load_seq_ml(test_path, "test", test_neg_num, seq_len, contain_user=True)
    fea_cols = {
        'user_num': user_num,
        'item_num': item_num,
        'seq_len': seq_len,
        'embed_dim': 64
    }
    params = {
        'fea_cols': fea_cols,
        'mode': 'inner',
        'loss_name': 'hinge_loss',
        'gamma': 0.5,
        'w': 0.5,
        'embed_reg': 0.
    }
    model = AttRec(**params)
    model.summary()
    model.compile(optimizer=Adam(learning_rate=learning_rate))
    for epoch in range(1, epochs+1):
        t1 = time()
        model.fit(
            x=train_data,
            validation_data=val_data,
            epochs=1,
            batch_size=batch_size
        )
        eval_dict = eval_pos_neg(model, test_data, batch_size, ["hr", "ndcg"], k)
        t2 = time()
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, NDCG = %.4f, '
              % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['ndcg']))


main()