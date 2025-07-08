"""Train LINE node embeddings.
"""

import argparse
import logging
import os
import time

os.environ["KERAS_BACKEND"] = "jax"

import keras
import polars as pl

from line import get_embeddings, LineGraphDataset, LineLoss, LineModel


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train LINE node embeddings.",
        epilog="""Example: python train_line.py \
            --edgelist-filename /data/processed/flat_2021_edgelist.parquet \
            --adjacency-filename /data/processed/flat_2021_adjacency_list.parquet
        """
    )

    parser.add_argument("--edgelist-filename", required=True, help="Filename of edgelist in Parquet format.")
    parser.add_argument("--adjacency-filename", required=True, help="Filename of adjacency list in Parquet format.")
    parser.add_argument("--embedding-dim", default=32, type=int, help="Number of embedding dimensions.")
    parser.add_argument("--batch-size", default=10_000, type=int, help="Number of edges to process in a single batch.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of times the entire edgelist is iterated over.")
    parser.add_argument("--order", default="all", help="Order of the embeddings. Can be 'first', 'second', or 'all'.")
    parser.add_argument("--negative-ratio", default=5, type=int, help="Ratio of edges for negative sampling.")
    parser.add_argument("--learning-rate", default=0.025, type=float, help="Learning rate.")
    parser.add_argument("--verbose", action="store_true", help="Enable keras logging.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility.")
    parser.add_argument("--logdir", default="logs", type=str, help="Directory for logging.")

    return parser.parse_args(args)


logger = logging.getLogger(__name__)


def main(args=None):
    args = cmdline_parse_args(args)
    
    out_basename = f"line_{os.path.basename(args.edgelist_filename)}_dim{args.embedding_dim}"

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join(args.logdir, out_basename),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO
    )

    out_dirname = os.path.join("data", "models", out_basename)

    os.makedirs(out_dirname, exist_ok=True)

    start_time = time.time()

    dataset = LineGraphDataset(
        args.edgelist_filename,
        args.adjacency_filename,
        args.batch_size,
        args.negative_ratio,
        two_outputs=args.order == "all",
        seed = args.seed
    )

    model = LineModel(dataset.num_nodes, embedding_dim=args.embedding_dim, order=args.order)

    model.compile(optimizer=keras.optimizers.Adam(
        learning_rate=keras.optimizers.schedules.InverseTimeDecay(
            initial_learning_rate=args.learning_rate,
            decay_steps=len(dataset) * args.epochs,
            decay_rate=1
        )
    ), loss=LineLoss())

    history = model.fit(
        dataset.generate(args.epochs),
        steps_per_epoch=len(dataset),
        epochs=args.epochs,
        callbacks=[keras.callbacks.ModelCheckpoint(filepath=os.path.join(out_dirname, "line_model.keras"))],
        verbose=args.verbose
    )

    logger.info("Finished training in %s hours", round((time.time() - start_time) / 3600, 2))

    pl.DataFrame(history.history).write_csv(os.path.join(out_dirname, "line_training_loss.csv"))

    embeddings = get_embeddings(model, dataset.idx2node)

    embeddings.save(os.path.join(out_dirname, "embeddings.kv"))


if __name__ == "__main__":
    main()
    