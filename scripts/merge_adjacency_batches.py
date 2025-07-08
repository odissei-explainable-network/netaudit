"""Merge adjacency list batches into single adjacency list.
"""

import argparse
import logging
import os
import time

import polars as pl


logger = logging.getLogger(__name__)


def merge_adjacency_batches(adjacency_list_dir, adacency_list_filename, streaming):
    (pl.scan_parquet(os.path.join(adjacency_list_dir, "*.parquet")) # Fetch all batches
        .group_by("RINPERSOON") # For each unique source
        .agg(pl.concat_list("neighbors").flatten().unique().sort()) # Find all unique neighbors and sort them
        .sort(by="RINPERSOON") # Sort by source
        .collect(streaming=streaming)
        .write_parquet(adacency_list_filename)
    )


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Merge adjacency list batches into single adjacency list.",
        epilog="""Example: python merge_adjacency_batches.py \
            --adjacency-dir /data/processed/flat_2021_adjacency_batches \
            --adjacency-filename /data/processed/flat_2021_adjacency_list.parquet
        """
    )

    parser.add_argument("--adjacency-dir", required=True, help="Directory with adjacency list batches.")
    parser.add_argument("--adjacency-filename", required=True, help="Filename of merged adjacency list in Parquet format.")
    parser.add_argument("--no-streaming", action="store_false", help="Disable merging using streaming.")
    return parser.parse_args(args)


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "merge_adjacency_batches.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO
    )
    start_time = time.time()
    logger.info("Starting merging batches")
    merge_adjacency_batches(args.adjacency_dir, args.adjacency_filename, args.no_streaming)
    logger.info("Finished merging batches in %s h", round((time.time() - start_time)/3600, 5))


if __name__ == '__main__':
	main()
        