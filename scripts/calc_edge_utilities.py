"""Compute edge utilities from node embeddings in target dimension.
"""

import argparse
import logging
import os
import time

import numpy as np
import polars as pl

from gensim.models import KeyedVectors
from numba.typed import Dict, List
from numba.core import types

from edge_utilities import calc_edge_interpretability_numba

logger = logging.getLogger(__name__)


def convert_numba_to_python(obj):
    if isinstance(obj, Dict):
        return {k: convert_numba_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, List):
        return [convert_numba_to_python(x) for x in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Compute edge utilities from node embeddings in target dimension.",
        epilog="""Example: python calc_interpretability.py \
            --adjacency-filename /data/processed/flat_2021_adjacency_list.parquet \
            --embeddings-filename /data/models/flat_2021/embeddings.kv \
            --utilities-filename /data/processed/flat_2021_utilities_dim1.parquet \
            --dimension 1
"""
    )
    parser.add_argument("--adjacency-filename", required=True, help="Filename of adjacency list in Parquet format.")
    parser.add_argument("--embeddings-filename", required=True, help="Filename of node embeddings in gensim embeddings vector format with '.kv' ending.")
    parser.add_argument("--utilities-filename", required=True, help="Filename of edge utilities in Parquet format.")
    parser.add_argument("--dimension", required=True, type=int, help="Target embedding dimension for which to compute edge utilities.")

    args = parser.parse_args(args)

    return args


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "calc_interpretability.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
        encoding="utf-8"
    )

    logger.info("Loading embeddings from %s", args.embeddings_filename)
    
    wv_embeddings = KeyedVectors.load(args.embeddings_filename)

    logger.info("Reading adjacency list from %s", args.adjacency_filename)
    df = pl.read_parquet(args.adjacency_filename)
    logger.info("Loaded adjacency list into memory")

    nodes = np.asarray(df["RINPERSOON"], dtype=np.int64)

    adjacency_dict_numba = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )

    for k, v in zip(nodes, df["neighbors"]):
        k = types.int64(k)
        adjacency_dict_numba[k] = np.asarray(v, dtype=np.int64)
    
    logger.info("Created adjacency numba dictionary")
    
    logger.info("Creating embeddings numba dictionary")

    embeddings_dict_numba = Dict.empty(
        key_type=types.int64,
        value_type=types.float32[:]
    )

    for node in wv_embeddings.index_to_key:
        k = types.int64(node)
        embeddings_dict_numba[k] = np.asarray(wv_embeddings[node], dtype=np.float32)

    logger.info("Created embeddings numba dictionary")

    start_time = time.time()

    edge_utilities = calc_edge_interpretability_numba(adjacency_dict_numba, embeddings_dict_numba, nodes, args.dimension)

    logger.info("Finished interpretability in %s seconds", round((time.time() - start_time) / 3600, 2))

    logger.info("Saving edge utilities to %s", args.utilities_filename)
    pl.DataFrame({"RINPERSOON": nodes, "neighbor_utilities": edge_utilities}).write_parquet(args.utilities_filename)


if __name__ == '__main__':
    main()
    