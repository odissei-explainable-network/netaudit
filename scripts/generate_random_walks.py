"""Generate random walks from an adjacency matrix.
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

from random_walks import random_walks

logger = logging.getLogger(__name__)


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate random walks from an adjacency matrix.",
        epilog="""Example: python generate_random_walks.py \
            --adjacency-filename /data/processed/flat_2021_adjacency_list.parquet \
            --walks-dirname /data/walks/flat_2021
        """
    )

    parser.add_argument("--adjacency-filename", required=True, help="Filename of adjacency list in Parquet format.")
    parser.add_argument("--walks-dirname", required=True, help="Directory for storing random walks in numpy format.")
    parser.add_argument("--walk-length", type=int, default=10, help="Length of random walks.")
    parser.add_argument("--num-walks", type=int, default=10, help="Number of walks for each node.")
    parser.add_argument("--seed", type=int, default=2024, help="Seed for reproducibility.")
    return parser.parse_args(args)


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "generate_random_walks.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO
    )

    logger.info("Reading adjacency list from %s", args.adjacency_filename)
    df = pl.read_parquet(args.adjacency_filename)
    logger.info("Loaded adjacency list into memory")

    logger.info("Creating adjacency numba dictionary")
    adjacency_dict_numba = Dict.empty(
        key_type=types.int64,
        value_type=types.int64[:]
    )

    for k, v in zip(df["RINPERSOON"], df["neighbors"]):
        k = types.int64(k)
        adjacency_dict_numba[k] = np.asarray(v, dtype=np.int64)
    
    logger.info("Created adjacency numba dictionary")

    nodes = np.asarray(df["RINPERSOON"], dtype=np.int64)

    os.makedirs(args.walks_dirname, exist_ok=True)

    set_seed_numba(args.seed)

    for i in range(args.num_walks):
        logger.info("Generating random walks file %s", i)
        walks = random_walks(adjacency_dict_numba, nodes, walk_length=args.walk_length)
        out_filename = os.path.join(args.walks_dirname, f"shard_{i}.npy")
        logger.info("Saving walks to file %s", out_filename)
        np.save(out_filename, walks)

    logger.info("Finished generating random walks")


if __name__ == '__main__':
	main()