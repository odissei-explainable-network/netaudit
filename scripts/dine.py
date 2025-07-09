"""Implementation of DINE: Dimensional Interpretability of Node Embeddings.

Method implemented by Piaggesi, S., Khosla, M., Panisson, A, & Anand, A. (2024).
DINE: Dimensional interpretability of node embeddings.
IEEE Transactions on Knowledge and Data Engineering. http://arxiv.org/abs/2310.01162

Implementation adapted from: simonepiaggesi @ https://github.com/simonepiaggesi/dine

"""

import os
import logging

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np

from gensim.models import KeyedVectors

EPS = 1e-15


logger = logging.getLogger(__name__)


class DineDataset:
    def __init__(self, embeddings_filename, batch_size, noise_level, shuffle, seed):
        self.embeddings_filename = embeddings_filename
        self.batch_size = batch_size
        self.noise_level = noise_level
        self.seed = seed

        self.keyed_vectors = KeyedVectors.load(self.embeddings_filename)
        self.node_names = self.keyed_vectors.index_to_key
        self.num_batches = len(self.node_names) // batch_size + 1
        self.embedding_dim = self.keyed_vectors[self.node_names[0]].shape[0]

        if shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.node_names)
            self.rng = rng


    def sample_noise(self, batch_shape):
        return self.rng.normal(loc=np.zeros(self.embedding_dim), scale=np.full(self.embedding_dim, self.noise_level), size=(batch_shape, self.embedding_dim))


    def generate(self, epochs):
        for _ in range(epochs):
            for i in range(self.num_batches):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, len(self.node_names))
                batch_node_names = self.node_names[start_idx:end_idx]
                batch_embeddings_x = self.keyed_vectors[batch_node_names]
                batch_embeddings_y = np.copy(batch_embeddings_x)
                
                if self.noise_level is not None:
                    batch_embeddings_x = batch_embeddings_x + self.sample_noise(batch_embeddings_x.shape[0])

                yield batch_embeddings_x, batch_embeddings_y


class DineModel(keras.Model):
    def __init__(self, input_dim, embedding_dim, lambda_size, lambda_orth):
        super().__init__()

        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.lambda_size = lambda_size
        self.lambda_orth = lambda_orth

        self.linear_1 = keras.layers.Dense(self.embedding_dim)
        self.linear_2 = keras.layers.Dense(self.embedding_dim)

        self.reconstruction_loss_fun = keras.losses.MeanSquaredError()
        self.size_loss = None
        self.orth_loss = None
        

    def _calc_size_loss(self, mask):   
        mask_size = keras.ops.sum(mask, axis=list(range(1, len(mask.shape))))
        mask_norm = mask_size / keras.ops.sum(mask_size, axis=0)
        mask_ent = keras.ops.sum(- mask_norm * keras.ops.log(mask_norm + EPS), axis=0)

        max_ent = keras.ops.log(mask.shape[0])

        return max_ent - keras.ops.mean(mask_ent)


    def _calc_orthogonality_loss(self, P):
        O = P @ P.T
        I = keras.ops.eye(O.shape[0])

        return self.reconstruction_loss_fun(O/keras.ops.norm(O), I/keras.ops.norm(I))
    

    def call(self, batch_x):
        linear_1_out = self.linear_1(batch_x)

        h = keras.ops.sigmoid(linear_1_out)

        out = self.linear_2(h)
                
        partitions = (h * h.sum(axis=0))

        size_loss = self._calc_size_loss(h.T) # size loss
        orth_loss = self._calc_orthogonality_loss(partitions.T) # orthogonality loss

        self.add_loss(self.lambda_size * size_loss + self.lambda_orth * orth_loss)
        
        return out
    

    def get_embedding(self, batch_x):
        linear_1_out = self.linear_1(batch_x)

        return keras.ops.sigmoid(linear_1_out)
