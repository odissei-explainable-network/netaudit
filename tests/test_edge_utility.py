"""Unit tests for edge utility calculation.
"""

import numpy as np
import pytest

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


def test_edge_utility_neutral():
    source_embedding = np.array([1.0, 1.0, 0.1, 1.0])
    target_embedding = np.array([0.1, 0.1, 1.0, 0.1])

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(2, edge_embedding)

    assert edge_utility == pytest.approx(0.0)


def test_edge_utility_weird():
    source_embedding = np.array([0.2, 0.2, 0.1, 0.2])
    target_embedding = np.array([0.2, 0.2, 0.1, 0.2])

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(2, edge_embedding)

    assert edge_utility < 0


def test_edge_utility_weird_2():
    source_embedding = np.array([0.2, 0.2, 0.3, 0.2])
    target_embedding = np.array([0.2, 0.2, 0.3, 0.2])

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(2, edge_embedding)

    assert edge_utility < 0


def test_edge_utility_weird_3():
    source_embedding = np.array([0.2, 0.25, 0.15, 0.2])
    target_embedding = np.array([0.25, 0.2, 0.2, 0.25])

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(2, edge_embedding)

    assert edge_utility < 0


def test_cosine_similarity():
    def calc_cosine_similarity(source_embedding, target_embedding):
        return np.dot(source_embedding, target_embedding) # / (np.linalg.norm(source_embedding) * np.linalg.norm(target_embedding))
    
    source_embedding = np.array([0.8, 0.5, 0.8, 0.5])
    target_embedding = np.array([0.8, 0.5, 0.9, 0.5])

    similarity = calc_cosine_similarity(source_embedding, target_embedding)

    dim_slice = np.concatenate((np.arange(0,2), np.arange(2+1, len(source_embedding))))

    similarity_removed = calc_cosine_similarity(source_embedding[dim_slice], target_embedding[dim_slice])
    similarity_diff = similarity - similarity_removed

    assert similarity_diff < 0


def test_edge_utility_real():
    num_dim = 32
    source_embedding = np.array([0.5] * num_dim)
    target_embedding = np.array([0.5] * num_dim)
    target_dim = 2
    source_embedding[target_dim] = 0.1
    target_embedding[target_dim] = 0.1

    edge_embedding = source_embedding * target_embedding

    edge_utility = calc_dimension_utility(target_dim, edge_embedding)

    assert edge_utility < 0


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
