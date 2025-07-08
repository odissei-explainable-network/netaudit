"""Create an flattened edgelist in Parquet format from multiple edgelists in CSV format.
"""

import argparse
import logging
import os
import time

import polars as pl


logger = logging.getLogger(__name__)


def process_batch(batch):
    # Creates batch of undirected edges
    reversed_batch = batch.select([
        pl.col("RINPERSOONRELATIE").alias("RINPERSOON"),
        pl.col("RINPERSOON").alias("RINPERSOONRELATIE"),
    ])
     
    undirected_batch = pl.concat([batch, reversed_batch])

    # Removes self-connections
    return undirected_batch.filter(pl.col("RINPERSOON") != pl.col("RINPERSOONRELATIE"))


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Create an flattened edgelist in Parquet format from multiple edgelists in CSV format.",
        epilog="""Example: python create_edgelist_from_csv \
    --csv-string /data/raw/*NETWERK2020TAB.csv \
    --edgelist-filename /data/processed/flat_2021_edgelist.parquet"""
    )

    parser.add_argument("--csv-string", required=True, help="String of edgelists in CSV format. Supports glob patterns.")
    parser.add_argument("--edgelist-filename", required=True, help="Filename of flattened edgelist in Parquet format.")
    return parser.parse_args(args)


def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "create_edgelist_from_csv.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO
    )

    logger.info("Scanning CSV files from %s", args.csv_string)
    (pl.scan_csv(args.csv_string, separator=";", schema_overrides={
            "RINPERSOONS": pl.String,
            "RINPERSOON": pl.Int64,
            "RINPERSOONSRELATIE": pl.String,
            "RINPERSOONRELATIE": pl.Int64, 
            "RELATIE": pl.Int32,
        }, low_memory=True)
        .select(pl.col("RINPERSOON", "RINPERSOONRELATIE"))
        .pipe(process_batch)
        .unique() # Retains unique edges
        .sink_parquet(args.edgelist_filename)
    )

    logger.info("Wrote edgelist to %s", args.edgelist_filename)


if __name__ == '__main__':
	main()
        