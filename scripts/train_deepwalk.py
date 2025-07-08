"""Train DeepWalk node embeddings and generate random walks on-the-fly.
"""

import argparse
import logging
import os
import time

import numba

import numpy as np
import polars as pl

from gensim.models.callbacks import CallbackAny2Vec
from gensim.models import Word2Vec

from numba import njit
from numba.typed import Dict
from numba.core import types

from random_walks import GraphRandomWalksLoading, GraphRandomWalksStreaming


logger = logging.getLogger(__name__)
gensim_logger = logging.getLogger("gensim.models.word2vec")


class DeepWalkCallback(CallbackAny2Vec):
    """Custom callback for tracking loss in Skip-Gram model

    Be very careful when interpreting the loss because there is a bug in gensim when it comes to tracking the loss:
    https://github.com/piskvorky/gensim/issues/2735

    """
    epoch = 0
    loss = 0


    def __init__(self, filename):
        self.filename = filename


    def on_train_begin(self, model):
        logger.info("Starting training")

    
    def on_train_end(self, model):
        logger.info("Finished training")


    def on_epoch_end(self, model):
        cum_loss = model.get_latest_training_loss()
        current_loss = cum_loss-self.loss
        logger.info("Epoch: %s \t Training loss: %s", self.epoch, current_loss)

        with open(self.filename, "a") as file:
            file.write(f"{self.epoch}\t{current_loss}\n")

        self.loss += current_loss
        self.epoch += 1


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train DeepWalk node embeddings and generate random walks on-the-fly.",
        epilog="""Example: python train_deepwalk.py \
            --adjacency-filename /data/processed/flat_2021_adjacency_list.parquet \
            --epochs 1
        """
    )

    parser.add_argument("--adjacency-filename", required=True, help="Filename of adjacency list in Parquet format.")

    parser.add_argument("--walks-dirname", default=None, help="Load random walks from disk.")

    parser.add_argument("--load-model", action="store_true", help="Load an existing model and continue training.")

    parser.add_argument('--embedding-dim', default=32, type=int,
                        help='Number of latent dimensions to learn for each node.')

    parser.add_argument("--walk-length", type=int, default=10, help="Length of each random walk.")

    parser.add_argument('--window-size', default=5, type=int,
                        help='Window size of SkipGram model.')
    
    parser.add_argument('--epochs', default=20, type=int,
                        help='How many times to iterate over each walk.')
    
    parser.add_argument("--num-walks", default=10, type=int, help="Number of walks to generate for each node.")

    parser.add_argument('--workers', default=8, type=int,
                        help='Number of parallel processes.')

    parser.add_argument('--seed', default=42, type=int,
                        help='Seed for random walk generator.')
    
    parser.add_argument("--logdir", default="logs", type=str, help="Directory for logging.")

    return parser.parse_args(args)


def main(args=None):
    args = cmdline_parse_args(args)

    base_split = "_".join(args.adjacency_filename.split("_")[:-2])

    out_basename = f"{os.path.basename(base_split)}_len{args.walk_length}_window{args.window_size}_walks{args.num_walks}_dim{args.embedding_dim}"

    file_handler = logging.FileHandler(
        os.path.join(args.logdir, out_basename),
        mode="w"
    )
    
    # Configure logging
    formatter = logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
    level = logging.INFO

    logger.setLevel(level)
    gensim_logger.setLevel(level)
    
    file_handler.setFormatter(formatter)
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(level)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    gensim_logger.addHandler(file_handler)
    gensim_logger.addHandler(stream_handler)

    if (args.walks_dirname is not None):
        logger.info("Loading random walks into memory from %s", args.walks_dirname)
        walks = GraphRandomWalksLoading(args.walks_dirname, args.num_walks, args.walk_length)
    else:
        logger.info("Using %s to generate random walks", args.adjacency_filename)
        walks = GraphRandomWalksStreaming(args.adjacency_filename, args.num_walks, args.walk_length)

    # Save embeddings to new subdirectory in data/models
    out_dirname = os.path.join("data", "models", out_basename)

    os.makedirs(out_dirname, exist_ok=True)

    # Save in gensim native format
    model_filename = os.path.join(out_dirname, "embeddings.kv")

    # Write loss to text file
    callbacks=[DeepWalkCallback(os.path.join(out_dirname, "training_loss.txt"))]

    start_time = time.time()

    if args.load_model:
        logger.info("Loading model from %s and continue training", model_filename)
        model = Word2Vec.load(model_filename)
        model.train(walks, total_examples=model.corpus_count, epochs=args.epochs, compute_loss=model.compute_loss, callbacks=callbacks)
    else:
        logger.info("Training Skip-Gram model")
        model = Word2Vec(walks, vector_size=args.embedding_dim, window=args.window_size, min_count=0, sg=1, workers=args.workers, epochs=args.epochs, seed=args.seed, compute_loss=True, callbacks=callbacks)
    
    logger.info("Finished training in %s hours", round((time.time() - start_time) / 3600, 2))

    logger.info("Saving embeddings")
    model.wv.save(model_filename)


if __name__ == '__main__':
    main()
        