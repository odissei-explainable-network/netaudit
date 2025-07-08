""" Implementation of the LINE: Large Scale Informed Network Embeddings.

Method developed by: Tang, J., Qu, M., Wang, M., Zhang, M., Yan, J., & Mei, Q. (2015).
LINE: Large-scale information network embedding.
Proceedings of the 24th International Conference on World Wide Web, 1067–1077.
https://doi.org/10.1145/2736277.2741093

Implementation adapted from: Weichen Shen @ https://github.com/shenweichen/GraphEmbedding

"""

import logging
import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import keras
import numpy as np
import polars as pl

import pyarrow.parquet as pq

from numba import njit

from gensim.models import KeyedVectors


class LineLoss(keras.losses.Loss):
    def call(self, y_true, y_pred):
        return -keras.ops.mean(keras.ops.log(keras.ops.sigmoid(y_true * y_pred)))

@njit(fastmath=True, cache=True, nogil=True)
def sample_alias_numba(size, n, idx, accept, alias, labels):
    idx = (np.array([np.random.uniform() for _ in range(size)]) * n).astype(np.int64)

    probs = np.array([np.random.uniform() for _ in range(size)], dtype=np.float64)
    
    out = np.where(probs < accept[idx], idx, alias[idx])

    return labels[out]


class AliasTable:
    def __init__(self, weights, labels=None):
        self.weights = np.array(weights)

        if labels is not None:
            self.labels = np.array(labels)
        else:
            self.labels = np.arange(len(weights))

        self.n = len(weights)

        self.accept, self.alias = self._create_table(self.weights, self.n)

    
    @staticmethod
    def _create_table(weights, n):
        accept, alias = [0] * n, [0] * n
        small, large = [], []
        weights_ = weights * n
        for i, prob in enumerate(weights_):
            if prob < 1.0:
                small.append(i)
            else:
                large.append(i)

        while small and large:
            small_idx, large_idx = small.pop(), large.pop()
            accept[small_idx] = weights_[small_idx]
            alias[small_idx] = large_idx
            weights_[large_idx] = weights_[large_idx] - (1 - weights_[small_idx])

            if weights_[large_idx] < 1.0:
                small.append(large_idx)
            else:
                large.append(large_idx)

        while large:
            large_idx = large.pop()
            accept[large_idx] = 1
        while small:
            small_idx = small.pop()
            accept[small_idx] = 1

        return np.array(accept), np.array(alias)

    
    def sample(self, size=1, idx=None, rng=None):
        return sample_alias_numba(size, self.n, idx, self.accept, self.alias, self.labels)


class LineModel(keras.Model):
    def __init__(self, num_nodes, embedding_dim=8, order="all", embedding_init="normal", name="line", **kwargs):
        super().__init__(name=name, **kwargs)
        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim
        self.order = order

        self.embedding_init = embedding_init

        lora_rank = None

        self.first_embedding = keras.layers.Embedding(
            self.num_nodes,
            embedding_dim,
            embeddings_initializer=self.embedding_init,
            name='embedding_1',
            lora_rank=lora_rank,
        )
        self.second_embedding = keras.layers.Embedding(
            self.num_nodes,
            embedding_dim,
            embeddings_initializer=self.embedding_init,
            name='embedding_2',
            lora_rank=lora_rank,
        )
        self.context_embedding = keras.layers.Embedding(
            self.num_nodes,
            embedding_dim,
            embeddings_initializer=self.embedding_init,
            name='embedding_context',
            lora_rank=lora_rank,
        )

        self.first_dot = keras.layers.Lambda(lambda x: keras.ops.sum(
            x[0] * x[1], axis=-1, keepdims=False), name='first_order')
        self.second_dot = keras.layers.Lambda(lambda x: keras.ops.sum(
            x[0] * x[1], axis=-1, keepdims=False), name='second_order')

        # self.first_dot = keras.layers.Dot(axes=-1)
        # self.second_dot = keras.layers.Dot(axes=-1)
        
    
    def call(self, inputs):
        # start_time = time.time()
        v_i = inputs[0]
        v_j = inputs[1]

        v_i_emb = self.first_embedding(v_i)
        v_j_emb = self.first_embedding(v_j)

        # print("First layer: " + str(round(time.time() - start_time, 2)))

        v_i_emb_second = self.second_embedding(v_i)

        # print("Second layer: " + str(round(time.time() - start_time, 2)))
        v_j_context_emb = self.context_embedding(v_j)
        # print("Context layer: " + str(round(time.time() - start_time, 2)))

        first = self.first_dot([v_i_emb, v_j_emb])
        second = self.second_dot([v_i_emb_second, v_j_context_emb])

        # print("Dot products: " + str(round(time.time() - start_time, 2)))

        if self.order == 'first':
            return [first]
        elif self.order == 'second':
            return [second]
        
        return [first, second]


class LineGraphDataset:
    def __init__(self, edgelist_filename, adjacency_filename, batch_size, negative_ratio=5, power=0.75, two_outputs=False, seed=None):
        self.edgelist_filename = edgelist_filename
        self.adjacency_filename = adjacency_filename
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.power = power
        self.two_outputs = two_outputs
        self.rng = np.random.default_rng(seed)
        self.logger = logging.getLogger("LineGraphDataset")

        self.load_data_from_parquet()
        self.preprocess_graph()

        self.pos_size = self.batch_size // (1 + negative_ratio)
        self.neg_size = self.pos_size * negative_ratio


    def load_data_from_parquet(self):
        self.logger.info("Reading edgelist from %s", self.edgelist_filename)
        self.edgelist = pq.ParquetFile(self.edgelist_filename)
        self.logger.info("Reading adjacency list from %s", self.adjacency_filename)
        self.adjacency_list = pl.read_parquet(self.adjacency_filename).sort("RINPERSOON")

        # if "RINPERSOON" not in self.edgelist.columns or "RINERPSOONRELATIE" not in self.edgelist.columns:
        #     raise ValueError("Edgelist must contain 'RINERPSOON' and 'RINPERSOONRELATIE' columns")
        

    def preprocess_graph(self):
        self.logger.info("Creating node index")
        self.idx2node = self.adjacency_list["RINPERSOON"].to_list()
        self.node2idx = {node: idx for idx, node in enumerate(self.idx2node)}
        self.num_nodes = len(self.idx2node)

        self.logger.info("Creating node alias table")
        degree = self.adjacency_list.select(pl.col("neighbors").list.len()).to_series().to_numpy()
        self.num_edges = np.sum(degree)
        out_degree_pow = degree ** self.power

        node_norm_prob = out_degree_pow / np.sum(out_degree_pow)

        self.node_alias_table = AliasTable(node_norm_prob)

        del self.adjacency_list


    def __len__(self):
        return int(self.num_edges*self.subsample/self.batch_size)


    def generate(self, epochs):
        self.logger.info("Collecting edges")
        self.logger.info("Finished collecting edges")
        self.logger.info("Starting processing batches")

        for i in range(epochs):
            for batch in self.edgelist.iter_batches(self.batch_size):
                batch = batch.to_pydict()
                batch_idx = np.array([(self.node2idx[s], self.node2idx[t]) for s, t in zip(batch["RINPERSOON"], batch["RINPERSOONRELATIE"]) if s in self.node2idx and t in self.node2idx])

                batch_source = batch_idx[:, 0]
                batch_target = batch_idx[:, 1]

                batch_neg = self.node_alias_table.sample(size=batch_idx.shape[0] * self.negative_ratio, rng=self.rng)

                batch_sign = np.ones(batch_idx.shape[0] * (1+self.negative_ratio))
                batch_sign[batch_idx.shape[0]:] = -1

                batch_x = [np.tile(batch_source, 1+self.negative_ratio), np.hstack([batch_target, batch_neg])]
                batch_y = [batch_sign]

                if self.two_outputs:
                    batch_y += [batch_sign]

                yield batch_x, batch_y


def get_embeddings(model, idx2node):

    idx = np.arange(len(idx2node))

    if model.order == 'first':
        embeddings = model.first_embedding.get_weights()[0]
    elif model.order == 'second':
        embeddings = model.second_embedding.get_weights()[0]
    else:
        embeddings = np.hstack((model.first_embedding.get_weights()[0][idx,:], model.second_embedding.get_weights()[0][idx,:]))

    obj = KeyedVectors(vector_size=embeddings.shape[1])

    obj.add_vectors(idx2node, embeddings[idx, :])

    return obj
    