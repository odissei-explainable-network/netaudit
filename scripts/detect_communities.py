"""Detect communities from an adjacency list using the Louvain algorithm.
"""

import argparse
import logging
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import polars as pl

from numba.typed import Dict
from numba.core import types

from louvain import find_best_partition


logger = logging.getLogger(__name__)


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Detect communities from an adjacency list using the Louvain algorithm.",
        epilog="""Example: python detect_communities.py \
        --adjacency-filename /data/processed/flat_2021_adjacency_list.parquet \
        --communities-filename /data/processed/flat_2021_communities.parquet
"""
    )
    parser.add_argument("--adjacency-filename", required=True, help="Filename of the adjacency list in Parquet format.")
    parser.add_argument("--communities-filename", required=True, help="Filename of the communities list in Parquet format. Each node in the adjacency list has a community ID.")

    args = parser.parse_args(args)

    return args


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "detect_communities.log"), # TODO: Add logname
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO,
        encoding="utf-8"
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

    logger.info("Detecting communities")

    start_time = time.time()

    partition, modularities = find_best_partition(adjacency_dict_numba, resolution=1.0, max_iter=-1, min_increase=1e-7)

    logger.info("Finished detecting communities in %s h", round((time.time() - start_time) / 3600.0, 5))

    pl.DataFrame({"RINPERSOON": np.array(list(adjacency_dict_numba.keys()))[np.array(list(partition.keys()))], "community": list(partition.values())}).write_parquet(args.communities_filename)

    logger.info("Modularities: %s", modularities)
    logger.info("Number of communities: %s", np.unique(list(partition.values())).shape[0])


if __name__ == '__main__':
    main()
