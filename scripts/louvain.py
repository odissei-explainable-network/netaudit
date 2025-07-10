"""Helper functions for community detection with the Louvain algorithm.

Code and implementation adapted from kerighan @ https://github.com/kerighan/louvain-numba

"""

import logging

import numba
import numpy as np

from numba import njit, prange
from numba.typed import Dict, List
from numba.core import types
from tqdm import tqdm

logger = logging.getLogger(__name__)

@njit(fastmath=True, cache=True, nogil=True)
def calc_modularity(node2com, internals, degrees, total_weight, resolution):
    result = 0.0

    for com in range(node2com.shape[0]):
        if degrees[com] > 0:
            result += resolution * internals[com] / total_weight
            result -= (degrees[com] / (2.0 * total_weight)) ** 2

    return result


def one_step(neighbors, weights, node2com, internals, loops, degrees, gdegrees, total_weight, resolution):
    num_nodes = len(neighbors)
    m = 2 * total_weight

    best_increase = -np.inf
    modified = False

    for node in range(num_nodes):
        node_com = node2com[node]
        node_com = types.int64(node_com)

        neighbor_weight = Dict.empty(key_type=types.int64, value_type=types.float64)

        for i in range(len(neighbors[node])):
            if neighbors[node][i] != node:
                neighborcom = node2com[neighbors[node][i]]
                if neighborcom not in neighbor_weight:
                    neighbor_weight[neighborcom] = weights[node][i]
                else:
                    neighbor_weight[neighborcom] += weights[node][i]
        
        node_gdegree = gdegrees[node]
        node_loop = loops[node]
        com_weight = neighbor_weight.get(node_com, 0.0)

        degrees[node_com] -= node_gdegree
        internals -= com_weight + node_loop
        node2com[node] = -1

        for com in neighbor_weight:
            weight = neighbor_weight[com]
            if weight <= 0:
                continue

            increase = resolution * weight - degrees[com] * node_gdegree / m
            if increase > best_increase:
                best_increase = increase
                best_move_node = node
                best_move_com = com
                best_neighbor_weight = neighbor_weight
                old_move_com = node_com
                modified = True

        node2com[node] = node_com
        degrees[node_com] += node_gdegree
        internals += com_weight + node_loop

        if not modified:
            return
        
        if old_move_com not in degrees:
            degrees[old_move_com] = 0.0
        if old_move_com not in internals:
            internals[old_move_com] = 0.0
        if old_move_com not in gdegrees:
            gdegrees[old_move_com] = 0.0
        if old_move_com not in loops:
            loops[old_move_com] = 0.0

        node2com[best_move_node] = -1
        degrees[old_move_com] -= gdegrees[best_move_node]
        internals[old_move_com] -= (best_neighbor_weight.get(old_move_com, 0.0) + loops[best_move_node])

        node2com[best_move_node] = best_move_com
        degrees[best_move_com] += gdegrees[best_move_node]
        internals[best_move_com] += (best_neighbor_weight[best_move_com] + loops[best_move_node])

@njit(fastmath=True, cache=True, nogil=True)
def find_neighbor_weights(node, neighbors, weights, node2com):
    neighbor_weight = Dict.empty(key_type=types.int64, value_type=types.float64)

    for i in range(len(neighbors[node])):
        if neighbors[node][i] != node:
            neighbor_com = node2com[neighbors[node][i]]
            if neighbor_com not in neighbor_weight:
                neighbor_weight[neighbor_com] = weights[node][i]
            else:
                neighbor_weight[neighbor_com] += weights[node][i]
    
    return neighbor_weight

@njit(fastmath=True, cache=True, nogil=True)
def one_level(num_nodes, node2com, internals, loops, degrees, gdegrees, neighbors, weights, total_weight, resolution, max_iter, min_increase):
    modified = True
    counter_iter = 0
    current_modularity = calc_modularity(node2com, internals, degrees, total_weight, resolution)
    new_modularity = current_modularity

    while modified and counter_iter != max_iter:
        # logger.info("Level iteration %s", counter_iter)
        current_modularity = new_modularity
        modified = False
        counter_iter += 1

        for node in range(num_nodes):
            node_com = node2com[node]
            neighbor_weight = find_neighbor_weights(node, neighbors, weights, node2com)

            # Remove node
            node2com[node] = -1
            degrees[node_com] -= gdegrees[node]
            internals[node_com] -= neighbor_weight.get(node_com, 0.0) + loops[node]

            best_com = node_com
            best_increase = 0

            for com in neighbor_weight:
                weight = neighbor_weight[com]
                increase = resolution * weight - (degrees[com] * gdegrees[node]) / (2.0 * total_weight)

                if increase > best_increase:
                    best_increase = increase
                    best_com = com

            # Insert node
            node2com[node] = best_com
            degrees[best_com] += gdegrees[node]
            internals[best_com] += neighbor_weight.get(best_com, 0.0) + loops[node]

            if best_com != node_com:
                modified = True

        new_modularity = calc_modularity(node2com, internals, degrees, total_weight, resolution)

        if new_modularity - current_modularity < min_increase:
            break

@njit(fastmath=True, cache=True, nogil=True)
def init_status(num_nodes, neighbors, weights):
    node2com = np.arange(num_nodes, dtype=np.int64)
    internals = np.zeros(num_nodes, dtype=np.float64)
    loops = np.zeros(num_nodes, dtype=np.float64)
    degrees = np.zeros(num_nodes, dtype=np.float64)
    gdegrees = np.zeros(num_nodes, dtype=np.float64)
    total_weight = 0.0

    for node in range(num_nodes):
        for i in range(len(neighbors[node])):
            neighbor = neighbors[node][i]
            neighbor_weight = weights[node][i]

            if neighbor == node:
                internals[node] += neighbor_weight
                loops[node] += neighbor_weight
                degrees[node] += 2.0 * neighbor_weight
                gdegrees[node] += 2.0 * neighbor_weight
                total_weight += 2.0 * neighbor_weight
            else:
                degrees[node] += neighbor_weight
                gdegrees[node] += neighbor_weight
                total_weight += neighbor_weight

    total_weight /= 2.0

    return node2com, internals, loops, degrees, gdegrees, total_weight

@njit(fastmath=True, cache=True, nogil=True)
def renumber_communities(node2com, num_nodes):
    com_num_nodes = np.zeros(num_nodes, dtype=np.int64)

    for node in range(num_nodes):
        com_num_nodes[node2com[node]] += 1

    com_new_index = np.zeros(num_nodes, dtype=np.int64)
    final_index = 0

    for com in range(num_nodes):
        if com_num_nodes[com] > 0:
            com_new_index[com] = final_index
            final_index += 1

    new_communities = List([List.empty_list(types.int64) for _ in range(final_index)])
    new_node2com = np.zeros(num_nodes, dtype=np.int64)

    for node in range(num_nodes):
        index = com_new_index[node2com[node]]
        new_communities[index].append(node)
        new_node2com[node] = index

    return new_communities, new_node2com

@njit(fastmath=True, cache=True, nogil=True)
def create_induced_graph(communities, neighbors, weights, node2com):
    new_num_nodes = len(communities)
    new_neighbors = List([List.empty_list(types.int64) for _ in range(new_num_nodes)])
    new_weights = List([List.empty_list(types.float64) for _ in range(new_num_nodes)])

    for com in range(new_num_nodes):
        to_insert = Dict.empty(key_type=types.int64, value_type=types.float64)

        for node in communities[com]:
            for i in range(len(neighbors[node])):
                neighbor = neighbors[node][i]
                neighbor_com = node2com[neighbor]
                neighbor_weight = weights[node][i]

                if neighbor_com not in to_insert:
                    to_insert[neighbor_com] = 0.0

                if neighbor == node:
                    to_insert[neighbor_com] += 2 * neighbor_weight
                else:
                    to_insert[neighbor_com] += neighbor_weight

        for neighbor_com, weight in to_insert.items():
            new_neighbors[com].append(neighbor_com)

            if neighbor_com == com:
                new_weights[com].append(weight / 2.0)
            else:
                new_weights[com].append(weight)

    return new_num_nodes, new_neighbors, new_weights

@njit(fastmath=True, cache=True, nogil=True)
def generate_dendrogram(adjacency_dict, num_nodes, resolution, max_iter, min_increase):
    node2idx = {node: idx for idx, node in enumerate(adjacency_dict.keys())}
    neighbors = [[node2idx[j] for j in neighbors] for neighbors in adjacency_dict.values()]
    weights = [[1.0] * len(n) for n in neighbors]

    node2com, internals, loops, degrees, gdegrees, total_weight = init_status(num_nodes, neighbors, weights)

    one_level(num_nodes, node2com, internals, loops, degrees, gdegrees, neighbors, weights, total_weight, resolution, max_iter, min_increase)

    new_modularity = calc_modularity(node2com, internals, degrees, total_weight, resolution)

    partition_list = []
    modularities = List.empty_list(types.float64)

    communities, node2com = renumber_communities(node2com, num_nodes)

    partition_list.append(node2com.copy())
    modularities.append(new_modularity)

    new_num_nodes, neighbors, weights = create_induced_graph(communities, neighbors, weights, node2com)

    node2com, internals, loops, degrees, gdegrees, total_weight = init_status(new_num_nodes, neighbors, weights)

    counter_iter = 0

    while True:
        # logger.info("Dendrogram iteration %s", counter_iter)
        counter_iter += 1

        one_level(new_num_nodes, node2com, internals, loops, degrees, gdegrees, neighbors, weights, total_weight, resolution, max_iter, min_increase)

        current_modularity = new_modularity

        new_modularity = calc_modularity(node2com, internals, degrees, total_weight, resolution)

        if new_modularity - current_modularity < min_increase:
            break

        communities, node2com = renumber_communities(node2com, new_num_nodes)

        partition_list.append(node2com.copy())
        modularities.append(new_modularity)

        new_num_nodes, neighbors, weights = create_induced_graph(communities, neighbors, weights, node2com)

        node2com, internals, loops, degrees, gdegrees, total_weight = init_status(new_num_nodes, neighbors, weights)

    return partition_list, modularities

@njit(fastmath=True, cache=True, nogil=True)
def find_best_partition(adjacency_dict, resolution, max_iter, min_increase):
    num_nodes = len(adjacency_dict)

    dendrogram, modularities = generate_dendrogram(adjacency_dict, num_nodes, resolution, max_iter, min_increase)

    partition = Dict.empty(key_type=types.int64, value_type=types.int64)

    for i in range(len(dendrogram[-1])):
        partition[i] = i

    for i in range(1, len(dendrogram) + 1):
        new_partition = Dict.empty(key_type=types.int64, value_type=types.int64)

        for j in range(len(dendrogram[-i])):
            new_partition[j] = partition[dendrogram[-i][j]]

        partition = new_partition

    return partition, modularities
