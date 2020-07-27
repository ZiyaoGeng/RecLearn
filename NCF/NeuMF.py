"""
Created on Nov 23, 2019

@author: GengZiyao (zggzy1996@gmail.com)
"""

from time import time

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers, optimizers

from GMF import GMF
from MLP import MLP
import configs
from DataSet import Dataset
from evaluate import evaluate_model
from utils import get_train_instances, load_pretrain_model

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class NeuMF(keras.Model):
    def __init__(self, num_users, num_items, mf_dim, layers, reg_layers, reg_mf):
        super(NeuMF, self).__init__()
        self.MF_Embedding_User = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=mf_dim,
            name='mf_embedding_user',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_mf[0]),
        )
        self.MF_Embedding_Item = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=mf_dim,
            name='mf_embedding_item',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_mf[1]),
        )
        self.MLP_Embedding_User = keras.layers.Embedding(
            input_dim=num_users,
            output_dim=int(layers[0] / 2),
            name='mlp_embedding_user',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_layers[0]),
        )
        self.MLP_Embedding_Item = keras.layers.Embedding(
            input_dim=num_items,
            output_dim=int(layers[0] / 2),
            name='mlp_embedding_item',
            embeddings_initializer='random_uniform',
            embeddings_regularizer=regularizers.l2(reg_layers[0]),
        )
        self.flatten = keras.layers.Flatten()
        self.mf_vector = keras.layers.Dot(axes=1)
        self.mlp_vector = keras.layers.Concatenate(axis=-1)
        self.layer1 = keras.layers.Dense(
            layers[1],
            name='layer1',
            activation='relu',
            kernel_regularizer=regularizers.l2(reg_layers[1]),
        )
        self.layer2 = keras.layers.Dense(
            layers[2],
            name='layer2',
            activation='relu',
            kernel_regularizer=regularizers.l2(reg_layers[2]),
        )
        self.layer3 = keras.layers.Dense(
            layers[3],
            name='layer3',
            activation='relu',
            kernel_regularizer=regularizers.l2(reg_layers[3]),
        )
        self.predict_vector = keras.layers.Concatenate(axis=-1)
        self.layer4 = keras.layers.Dense(
            1,
            activation='sigmoid',
            kernel_initializer='lecun_uniform',
            name='prediction'
        )

    @tf.function
    def call(self, inputs):
        # Embedding
        MF_Embedding_User = self.MF_Embedding_User(inputs[0])
        MF_Embedding_Item = self.MF_Embedding_Item(inputs[1])
        MLP_Embedding_User = self.MLP_Embedding_User(inputs[0])
        MLP_Embedding_Item = self.MLP_Embedding_Item(inputs[1])

        # MF
        mf_user_latent = self.flatten(MF_Embedding_User)
        mf_item_latent = self.flatten(MF_Embedding_Item)
        mf_vector = self.mf_vector([mf_user_latent, mf_item_latent])

        # MLP
        mlp_user_latent = self.flatten(MLP_Embedding_User)
        mlp_item_latent = self.flatten(MLP_Embedding_Item)
        mlp_vector = self.mlp_vector([mlp_user_latent, mlp_item_latent])
        mlp_vector = self.layer1(mlp_vector)
        mlp_vector = self.layer2(mlp_vector)
        mlp_vector = self.layer3(mlp_vector)

        # NeuMF
        vector = self.predict_vector([mf_vector, mlp_vector])
        output = self.layer4(vector)

        return output


if __name__ == '__main__':

    model_out_file = 'Save/%s_NeuMF_%d_%s_%d.h5' % (configs.dataset, configs.mf_dim, configs.layers, time())

    # --------------Loading data--------------
    t1 = time()
    dataset = Dataset('Data/%s' % configs.dataset)
    train, testRatings, testNegatives = dataset.trainMatrix, dataset.testRatings, dataset.testNegatives
    num_users, num_items = train.shape
    print('Load data done [%.1f s]. # user=%d, #item=%d, #train=%d, #test=%d'
          % (time()-t1, num_users, num_items, train.nnz, len(testRatings)))

    # --------------Build model---------------
    model = NeuMF(num_users, num_items, configs.mf_dim, configs.layers, configs.reg_layers, configs.reg_mf)
    model.build(input_shape=[1, 1])
    if configs.learner.lower() == "adagrad":
        model.compile(optimizer=optimizers.Adagrad(lr=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "rmsprop":
        model.compile(optimizer=optimizers.RMSprop(lr=configs.learning_rate), loss='binary_crossentropy')
    elif configs.learner.lower() == "adam":
        model.compile(optimizer=optimizers.Adam(lr=configs.learning_rate), loss='binary_crossentropy')
    else:
        model.compile(optimizer=optimizers.SGD(lr=configs.learning_rate), loss='binary_crossentropy')

    # -----------Load pretrain model-----------
    if configs.mf_pretrain != '' and configs.mlp_pretrain != '':
        gmf_model = GMF(num_users, num_items, configs.mf_dim)
        gmf_model.build(input_shape=([1, 1]))
        gmf_model.load_weights(configs.mf_pretrain)
        mlp_model = MLP(num_users, num_items, configs.layers, configs.reg_layers)
        mlp_model.build(input_shape=([1, 1]))
        mlp_model.load_weights(configs.mlp_pretrain)
        model = load_pretrain_model(model, gmf_model, mlp_model, len(configs.layers))

    # ---------------Init performance----------------
    (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, configs.topK)
    hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
    print('Init: HR = %.4f, NDCG = %.4f [%.1f s]' % (hr, ndcg, time() - t1))
    best_hr, best_ndcg, best_iter = hr, ndcg, -1
    if configs.out > 0:
        model.save_weights(model_out_file, overwrite=True)

    # -----------------Training model-------------
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

        # ----------------Evaluation--------------
        if epoch % configs.verbose == 0:
            (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, configs.topK)
            hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
            print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]'
                  % (epoch, t2 - t1, hr, ndcg, loss, time() - t2))
            if hr > best_hr:
                best_hr, best_ndcg, best_iter = hr, ndcg, epoch
                if configs.out > 0:
                    model.save_weights(model_out_file, overwrite=True)

    print('End. Best Iteration %d: HR = %.4f, NDCG = %.4f ' % (best_iter, best_hr, best_ndcg))
    if configs.out > 0:
        print('The best NeuMF model is saved to %s' % model_out_file)


