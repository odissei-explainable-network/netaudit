"""Helper functions for calculating edge utility as part of the DINE framework.

Method implemented by Piaggesi, S., Khosla, M., Panisson, A, & Anand, A. (2024).
DINE: Dimensional interpretability of node embeddings.
IEEE Transactions on Knowledge and Data Engineering. http://arxiv.org/abs/2310.01162

Implementation adapted from: simonepiaggesi @ https://github.com/simonepiaggesi/dine

"""

import logging

import numpy as np

from numba import njit, prange
from numba.typed import Dict
from numba.core import types


logger = logging.getLogger(__name__)

@njit
def calc_dimension_utility(target_dimension: int, edge_embedding: np.ndarray) -> np.ndarray:
    """Compute the utility of edge embeddings in a target dimension.
    """
    dim_slice = np.concatenate((np.arange(0, target_dimension), np.arange(target_dimension+1, len(edge_embedding))))
    edge_scores_removed = np.mean(edge_embedding[dim_slice])
    edge_scores_all = np.mean(edge_embedding)
    edge_scores_diff = edge_scores_all - edge_scores_removed

    edge_utility = edge_scores_diff

    return edge_utility # Only positive utility above threshold

@njit(parallel=True)
def calc_neighbor_utility(neighbors: np.ndarray, source_embedding: np.ndarray, embeddings: np.ndarray, target_dimension: int):
    """Compute edge utilities for all neighbors in target dimension.
    """
    num_neighbors = len(neighbors)
    edge_utility = np.zeros(num_neighbors, dtype=np.float64)

    for i in prange(num_neighbors):
        edge_embedding = source_embedding * embeddings[neighbors[i]]
        edge_utility[i] = calc_dimension_utility(target_dimension, edge_embedding)

    return edge_utility


nested_dict_type = types.ListType(types.DictType(types.unicode_type, types.float64))

dict_type = types.DictType(types.unicode_type, types.float64)

@njit
def calc_edge_interpretability_numba(adjacency_dict: Dict, embeddings: Dict, nodes: np.ndarray, target_dimension: int):
    """Compute edge utilities in target dimension for an adjacency list.
    """
    edge_utilities = []

    for j in range(len(nodes)):
        node = nodes[j]
        neighbors = adjacency_dict[node]

        source_embedding = embeddings[node]

        edge_utility = calc_neighbor_utility(neighbors, source_embedding, embeddings, target_dimension)

        edge_utilities.append(edge_utility) 

    return edge_utilities
