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


class MLP(keras.Model):
    def __init__(self, num_users, num_items, layers=[20, 10], reg_layers=[0, 0]):
        super(MLP, self).__init__()
        self.MLP_Embedding_User = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=int(layers[0] / 2),
            name='user_embedding',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_layers[0])
        )
        self.MLP_Embedding_Item = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=int(layers[0] / 2),
            name='item_embedding',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_layers[0]))
        self.flatten = keras.layers.Flatten()
        self.vector = keras.layers.Concatenate(axis=-1)
        self.layer1 = keras.layers.Dense(
            layers[1],
            activation='relu',
            name='layer1',
            kernel_regularizer=regularizers.l2(reg_layers[1]),
        )
        self.layer2 = keras.layers.Dense(
            layers[2],
            activation='relu',
            name='layer2',
            kernel_regularizer=regularizers.l2(reg_layers[2]),
        )
        self.layer3 = keras.layers.Dense(
            layers[3],
            activation='relu',
            name='layer3',
            kernel_regularizer=regularizers.l2(reg_layers[3]),
        )
        self.layer4 = keras.layers.Dense(
            1,
            name='prediction',
            activation='sigmoid',
            kernel_initializer='lecun_uniform'
        )

    @tf.function
    def call(self, inputs):
        # Embedding
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[1])

        # flatten
        user_latent = self.flatten(MLP_Embedding_User)
        item_latent = self.flatten(MLP_Embedding_Item)

        # concatenation of embedding layers
        x = self.vector([user_latent, item_latent])

        # MLP
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.layer4(x)

        return output


if __name__ == '__main__':

    model_out_file = 'Pretrain/%s_MLP_%s_%d.h5' % (configs.dataset, configs.layers, time())

    # --------------Loading data-------------
    t1 = time()
    dataset = Dataset('Data/%s' % configs.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print("Load data done [%.1f s]. #user=%d, #item=%d, #train=%d, #test=%d"
           % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # -------------Build model--------------
    model = MLP(num_users, num_items, configs.layers, configs.reg_layers)

    # -------------Compile model-------------
    if configs.learner.lower() == "adagrad":
        model.compile(optimizer=optimizers.Adagrad(lr=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "rmsprop":
        model.compile(optimizer=optimizers.RMSprop(lr=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(lr=configs.learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizers.SGD(lr=configs.learning_rate), loss='binary_crossentropy')

    """
    # --------------Init performance--------------
    t1 = time()
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, configs.topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - t1))
    """
    # --------------Train model-----------------
    # best_hr, best_ndcg, best_iter = hr, ndcg, -1
    for epoch in range(configs.epochs):
        t1 = time()

        # Generate training instances
        user_input, item_input, labels = get_train_instances(train, configs.num_negatives)
        print('Get_train_instances [%.1f s]' % (time() - t1))
        t1 = time()
        # --------------Fit model----------------
        hist = model.fit([np.array(user_input), np.array(item_input)],
                         np.array(labels),
                         batch_size=configs.batch_size,
                         epochs=1,
                         verbose=configs.verbose,
                         shuffle=True)
        t2 = time()
        """
        # --------------Evaluation--------------
        if epoch % configs.verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, configs.topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2-t1, hr, ndcg, loss, time()-t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if configs.out > 0:
                    model.save_weights(model_out_file, overwrite=True)
        """
    # print("End. Best Iteration %d: HR = %.4f, NDCG = %.4f. " % (best_iter, best_hr, best_ndcg))
    model.save_weights(model_out_file, overwrite=True)
    if configs.out > 0:
        print('The best MLP model is saved to %s' % model_out_file)