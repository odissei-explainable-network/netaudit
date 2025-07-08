"""Helper functions for generating random walks.
"""

import argparse
import logging
import os

import numba

import numpy as np
import polars as pl

from numba import njit
from numba.typed import Dict
from numba.core import types


logger = logging.getLogger(__name__)


@njit
def set_seed_numba(seed):
    np.random.seed(seed)


@njit
def random_choice(x):
    return np.random.choice(x)


@njit(parallel=True, nogil=True, fastmath=True)
def random_walks(adjacency_dict: numba.typed.Dict, nodes: numba.int64[:], walk_length: numba.int64 = 5):
    # Create 1 walk for every node in adjacency dict
    n_walks = len(nodes)
    # Initialize matrix for walks
    res = np.empty((n_walks, walk_length), dtype=np.int64)

    for i in numba.prange(n_walks):
        current_node = nodes[i]
        for j in range(walk_length):
            res[i, j] = current_node

            if current_node in adjacency_dict:
                neighbors = adjacency_dict[current_node]
            else:
                res[i, j] = -1 # Sanity check if neighbor current node is not in adjacency dict

            if len(neighbors) > 0:
                current_node = random_choice(neighbors)
            else:
                res[i, j] = -1

    return res


class GraphRandomWalksStreaming:
    """Helper class for creating random walks from the adjacency list on-the-fly.

    Implements an iterator that generates a new set of random walks for each node
    in the adjacency list on-the-fly everytime gensim.Word2Vec requests more data.
    This is much more disk-space efficient as it does not require the walks to be
    pre-generated.

    """
    def __init__(self, adjacency_filename, num_walks, walk_length):
        self.adjacency_filename = adjacency_filename
        self.num_walks = num_walks
        self.walk_length = walk_length

        logger.info("Reading adjacency list from %s", adjacency_filename)
        df = pl.read_parquet(adjacency_filename)
        logger.info("Loaded adjacency list into memory")

        logger.info("Creating adjacency numba dictionary")
        self.adjacency_dict_numba = Dict.empty(
            key_type=types.int64,
            value_type=types.int64[:]
        )

        for k, v in zip(df["RINPERSOON"], df["neighbors"]):
            k = types.int64(k)
            self.adjacency_dict_numba[k] = np.asarray(v, dtype=np.int64)
        
        logger.info("Created adjacency numba dictionary")

        self.nodes = np.asarray(df["RINPERSOON"], dtype=np.int64)

    
    def __iter__(self):
        for i in range(self.num_walks):
            logger.info("Generating walks shard %s", i)
            walks = random_walks(self.adjacency_dict_numba, self.nodes, walk_length=self.walk_length)

            if self.walk_length > walks.shape[1]:
                raise ValueError("Walk length cannot be higher than length of loaded walks")

            for walk in walks:
                # Omit erroneous nodes with id -1
                # gensim requries a list
                yield walk[walk > 0][:self.walk_length].tolist()


class GraphRandomWalksLoading:
    def __init__(self, dirname, num_walks, walk_length):
        self.dirname = dirname
        self.num_walks = num_walks
        self.walk_length = walk_length

    
    def __iter__(self):
        for i in range(self.num_walks):
            logger.info("Loading walks from shard %s", i)
            walks = np.load(os.path.join(self.dirname, f"shard_{i}.npy"))

            if self.walk_length > walks.shape[1]:
                raise ValueError("Walk length cannot be higher than length of loaded walks")

            for walk in walks:
                yield walk[walk > 0][:self.walk_length].tolist()
