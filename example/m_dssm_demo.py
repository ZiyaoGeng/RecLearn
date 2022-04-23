"""
Created on Apr 1, 2022
train DSSM demo
@author: Ziyao Geng(zggzy1996@163.com)
"""
import os
from absl import flags, app
from time import time
from tensorflow.keras.optimizers import Adam

from reclearn.models.matching import DSSM
from reclearn.data.datasets import movielens as ml
from reclearn.evaluator import eval_pos_neg

FLAGS = flags.FLAGS

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Setting training parameters
flags.DEFINE_string("file_path", "data/ml-1m/ratings.dat", "file path.")
flags.DEFINE_string("train_path", "data/ml-1m/ml_train.txt", "train path. If set to None, the program will split the dataset.")
flags.DEFINE_string("val_path", "data/ml-1m/ml_val.txt", "val path.")
flags.DEFINE_string("test_path", "data/ml-1m/ml_test.txt", "test path.")
flags.DEFINE_string("meta_path", "data/ml-1m/ml_meta.txt", "meta path.")
flags.DEFINE_integer("embed_dim", 64, "The size of embedding dimension.")
flags.DEFINE_float("embed_reg", 0.0, "The value of embedding regularization.")
flags.DEFINE_list("user_mlp", [128], "A list of user MLP hidden units.")
flags.DEFINE_list("item_mlp", [128], "A list of item MLP hidden units")
flags.DEFINE_string("activation", "relu", "Activation Name.")
flags.DEFINE_float("dnn_dropout", 0., "Float between 0 and 1. Dropout of user and item MLP layer.")
flags.DEFINE_boolean("use_l2norm", False, "Whether user embedding, item embedding should be normalized or not.")
flags.DEFINE_string("loss_name", "binary_cross_entropy_loss", "Loss Name.")
flags.DEFINE_float("gamma", 0.5, "If hinge_loss is selected as the loss function, you can specify the margin.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_integer("neg_num", 4, "The number of negative sample for each positive sample.")
flags.DEFINE_integer("epochs", 20, "train steps.")
flags.DEFINE_integer("batch_size", 512, "Batch Size.")
flags.DEFINE_integer("test_neg_num", 100, "The number of test negative samples.")
flags.DEFINE_integer("k", 10, "recall k items at test stage.")


def main(argv):
    # TODO: 1. Split Data
    if FLAGS.train_path == "None":
        train_path, val_path, test_path, meta_path = ml.split_data(file_path=FLAGS.file_path)
    else:
        train_path, val_path, test_path, meta_path = FLAGS.train_path, FLAGS.val_path, FLAGS.test_path, FLAGS.meta_path
    with open(meta_path) as f:
        max_user_num, max_item_num = [int(x) for x in f.readline().strip('\n').split('\t')]
    # TODO: 2. Load Data
    train_data = ml.load_data(train_path, FLAGS.neg_num, max_item_num)
    val_data = ml.load_data(val_path, FLAGS.neg_num, max_item_num)
    test_data = ml.load_data(test_path, FLAGS.test_neg_num, max_item_num)
    # TODO: 3. Set Model Hyper Parameters.
    model_params = {
        'user_num': max_user_num + 1,
        'item_num': max_item_num + 1,
        'embed_dim': FLAGS.embed_dim,
        'user_mlp': FLAGS.user_mlp,
        'item_mlp': FLAGS.item_mlp,
        'activation': FLAGS.activation,
        'dnn_dropout': FLAGS.dnn_dropout,
        'use_l2norm': FLAGS.use_l2norm,
        'loss_name': FLAGS.loss_name,
        'gamma': FLAGS.gamma,
        'embed_reg': FLAGS.embed_reg
    }
    # TODO: 4. Build Model
    model = DSSM(**model_params)
    model.compile(optimizer=Adam(learning_rate=FLAGS.learning_rate))
    # TODO: 5. Fit Model
    for epoch in range(1, FLAGS.epochs + 1):
        t1 = time()
        model.fit(
            x=train_data,
            epochs=1,
            validation_data=val_data,
            batch_size=FLAGS.batch_size
        )
        t2 = time()
        eval_dict = eval_pos_neg(model, test_data, ['hr', 'mrr', 'ndcg'], FLAGS.k, FLAGS.batch_size)
        print('Iteration %d Fit [%.1f s], Evaluate [%.1f s]: HR = %.4f, MRR = %.4f, NDCG = %.4f'
              % (epoch, t2 - t1, time() - t2, eval_dict['hr'], eval_dict['mrr'], eval_dict['ndcg']))


if __name__ == '__main__':
    app.run(main)