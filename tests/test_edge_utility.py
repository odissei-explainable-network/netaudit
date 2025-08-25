"""Unit tests for edge utility calculation.
"""

import numpy as np

from scripts.edge_utilities import calc_dimension_utility, calc_neighbor_utility


def test_edge_utility_negative():
    source_embedding = np.array([0.5, 0.5, 0.1, 0.5])
    target_embedding = np.array([0.5, 0.5, 1.0, 0.5])

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(2, edge_embedding)

    assert edge_utility < 0


def test_edge_utility_positive():
    source_embedding = np.array([1.0, 1.0, 0.5, 1.0])
    target_embedding = np.array([0.1, 0.1, 0.5, 0.1])

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(2, edge_embedding)

    assert edge_utility > 0


def test_neighbor_utility_negative():
    num_neighbors = 4
    neighbors = np.arange(num_neighbors)

    source_embedding = np.array([0.5, 0.5, 0.5, 0.5])

    target_embeddings = np.array([
        [0.5, 0.5, 0.1, 0.5],
        [0.5, 0.5, 0.2, 0.5],
        [0.5, 0.5, 0.3, 0.5],
        [0.5, 0.5, 0.4, 0.5]
    ])

    edge_utility_neighbors = calc_neighbor_utility(
        neighbors,
        source_embedding,
        target_embeddings,
        target_dimension=2
    )

    assert np.all(edge_utility_neighbors < 0)


def test_neighbor_utility_positive():
    num_neighbors = 4
    neighbors = np.arange(num_neighbors)

    source_embedding = np.array([0.5, 0.5, 0.5, 0.5])

    target_embeddings = np.array([
        [0.1, 0.1, 0.5, 0.1],
        [0.2, 0.2, 0.5, 0.2],
        [0.3, 0.3, 0.5, 0.3],
        [0.4, 0.4, 0.5, 0.4]
    ])

    edge_utility_neighbors = calc_neighbor_utility(
        neighbors,
        source_embedding,
        target_embeddings,
        target_dimension=2
    )

    assert np.all(edge_utility_neighbors > 0)
