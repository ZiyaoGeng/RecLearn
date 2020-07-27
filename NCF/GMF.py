"""
Created on Nov 22, 2019

@author: GengZiyao (zggzy1996@gmail.com)
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

from time import time
from DataSet import Dataset
from evaluate import evaluate_model
from utils import get_train_instances
import configs

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class GMF(keras.Model):
    def __init__(self, num_users, num_items, latent_dim, regs=[0, 0]):
        super(GMF, self).__init__()
        self.MF_Embedding_User = layers.Embedding(
            input_dim=num_users,
            output_dim=latent_dim,
            name='user_embedding',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(regs[0])
            )
        self.MF_Embedding_Item = layers.Embedding(
            input_dim=num_items,
            output_dim=latent_dim,
            name='item_embedding',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(regs[1]))
        self.flatten = layers.Flatten()
        self.predict_vector = layers.Dot(axes=1)
        self.dense = layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            name='prediction')

    @tf.function
    def call(self, inputs):
        # Embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[1])

        # flatten
        user_latent = self.flatten(MF_Embedding_User)
        item_latent = self.flatten(MF_Embedding_Item)

        # Element-wise product of user and item embeddings
        predict_vector = self.predict_vector([user_latent, item_latent])

        # Final prediction layer
        output = self.dense(predict_vector)

        return output


if __name__ == '__main__':

    model_out_file = 'Pretrain/%s_GMF_%d_%d.h5' % (configs.dataset, configs.num_factors, time())

    # ----------------Loading data---------------
    t1 = time()
    dataset = Dataset('Data/%s' % configs.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    print(train)
    num_users, num_items = train.shape

    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
          % (time() - t1, num_users, num_items, train.nnz, len(testRatings)))

    # --------------Build model-------------------
    model = GMF(num_users, num_items, configs.num_factors, configs.regs)

    # ---------------Compile model-----------------
    if configs.learner.lower() == "adagrad":
        model.compile(optimizer=optimizers.Adagrad(lr=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "rmsprop":
        model.compile(optimizer=optimizers.RMSprop(lr=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(lr=configs.learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizers.SGD(lr=configs.learning_rate), loss='binary_crossentropy')

    # --------------Init performance--------------
    t1 = time()
    # 消耗时间长
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, configs.topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f\t [%.1f s]' % (hr, ndcg, time() - t1))

    # ----------------Train model------------------
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(configs.epochs):
        t1 = time()
        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, configs.num_negatives)
        print('Get_train_instances [%.1f s]' % (time() - t1))
        t1 = time()
        # --------------Fit model-----------------
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=configs.batch_size,
                         epochs=1,
                         verbose=configs.verbose,
                         shuffle=True)
        t2 = time()

        # ----------------Evaluation--------------
        # 耗时长
        if epoch % configs.verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, configs.topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if configs.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    if configs.out > 0:
        print("The best GMF model is saved to %s" % model_out_file)