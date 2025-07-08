"""Transform an edgelist into adjacency list batches.
"""

import argparse
import logging
import os
import time

import polars as pl


logger = logging.getLogger(__name__)


def process_edge_list_batched(edgelist_filename, adjacency_list_dir, chunk_size=10000, directed=False):
    logger.info("Starting collecting edges")
    # Read edgelist from parquet file without loading entire file into memory
    lf = pl.scan_parquet(edgelist_filename) # set n_rows=N for debugging

    # Load edges into memory in chunks
    edge_chunks = lf.collect(streaming=True, chunk_size=chunk_size)

    logger.info("Finished collecting edges")
    logger.info("Starting processing batches")

    global_start_time = time.time()

    start_time = time.time()

    # Iterate over chunks and write chunk adjacency list into parquet partition
    for batch_num, chunk in enumerate(edge_chunks.iter_slices(chunk_size)):
        output_filename_batch = os.path.join(adjacency_list_dir, f"batch_{batch_num}.parquet")

        # Group by source nodes and store unique target nodes in list column
        (chunk
            .group_by("RINPERSOON")
            .agg(pl.col("RINPERSOONRELATIE")
                    .unique()
                    .alias("neighbors")
                )
            .write_parquet(output_filename_batch, compression="snappy")
        )

        logger.info("Batch: %s \t Total edges: %s \t Processing time: %s", batch_num, (batch_num+1)*chunk_size, round(time.time()-start_time, 3))
        
        start_time = time.time()

    logger.info("Finished processing chunks in %s hours", round((time.time() - global_start_time)/3600, 5))


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Transform an edgelist into adjacency list batches.",
        epilog="""Example: python create_adjacency_list_batched.py \
    --edgelist-filename /data/processed/flat_2021_edgelist.parquet \
    --outdir /data/processed/flat_2021_adjacency_batches"""
    )

    parser.add_argument("--edgelist-filename", required=True, help="Filename of edgelist to transform.")
    parser.add_argument("--outdir", required=True, help="Directory to store adjacency list batches.")
    parser.add_argument("--chunk-size", type=int, default=10_000_000, help="Number of edges in each adjacency list batch.")

    return parser.parse_args(args)


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "create_adjacency_list_batched.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO
    )

    os.makedirs(args.outdir, exist_ok=True)

    process_edge_list_batched(args.edgelist_filename, args.outdir, args.chunk_size, args.directed)


if __name__ == '__main__':
	main()
        