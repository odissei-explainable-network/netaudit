"""Train DINE regularizing auto-encoder and transform embeddings.
"""

import argparse
import logging
import os

os.environ["KERAS_BACKEND"] = "jax"

import hydra
import keras
import polars as pl

from gensim.models import KeyedVectors

from dine import DineDataset, DineModel


logger = logging.getLogger(__name__)


def optional_float(x):
    return None if x == "None" else float(x)


def cmdline_parse_args(args):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Train DINE regularizing auto-encoder and transform embeddings.",
        epilog="""Example: python train_dine.py --embeddings-filename /data/models/flat_2021/embeddings.kv
"""
    )
    parser.add_argument("--embeddings-filename", required=True, help="Filename of node embeddings in gensim embeddings vector format with '.kv' ending.")
    parser.add_argument('--noise-level', default=0.2, type=optional_float, help="Level of noise (standard deviation of isotropic Gaussian noise) to add to input embeddings.")
    parser.add_argument('--lambda-size', default=1.0, type=float, help="Weight of size loss in overall loss function.")
    parser.add_argument('--lambda-orth', default=1.0, type=float, help="Weight of orthogonality loss in overall loss function.")
    parser.add_argument('--learning-rate', default=0.1, type=float, help="Learning rate for auto-encoder.")
    parser.add_argument("--batch-size", default=10_000, type=int, help="Number of embedding vectors in a single batch.")
    parser.add_argument("--epochs", default=1, type=int, help="Number of times all embeddings are iterated over.")
    parser.add_argument("--verbose", action="store_true", help="Enable keras logging.")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle embeddings after each epoch.")
    parser.add_argument("--seed", default=42, type=int, help="Seed for reproducibility.")

    return parser.parse_args(args)

@hydra.main(version_base=None, config_path="../conf", config_name="config_dine")
def main(args=None):
    args = cmdline_parse_args(args)

    logging.basicConfig(
        handlers=[
            logging.FileHandler(
                os.path.join("logs", "train_dine.log"),
                mode="w"
            ),
            logging.StreamHandler()
        ],
        format="%(asctime)s - [%(levelname)s] - %(message)s",
        level=logging.INFO
    )

    logger.info("Loading DINE dataset")
    dataset = DineDataset(args.embeddings_filename, args.batch_size, args.noise_level, args.shuffle, args.seed)

    logger.info("Creating DINE model")
    model = DineModel(dataset.embedding_dim, dataset.embedding_dim, args.lambda_size, args.lambda_orth)

    model.compile(optimizer=keras.optimizers.Adam(learning_rate=args.learning_rate), loss=keras.losses.MeanSquaredError())

    logger.info("Starting training")
    history = model.fit(
        dataset.generate(args.epochs),
        steps_per_epoch=dataset.num_batches,
        epochs=args.epochs,
        callbacks=[
            keras.callbacks.ModelCheckpoint(filepath=os.path.join(os.path.dirname(args.embeddings_filename), "model_dine.keras"))
        ],
        verbose=args.verbose
    )

    logger.info("Finished training")
    pl.DataFrame(history.history).write_csv(os.path.join(os.path.dirname(args.embeddings_filename), "dine_training_loss.csv"))

    logger.info("Transforming embeddings")

    embeddings = KeyedVectors.load(args.embeddings_filename)

    transformed_embeddings = model.get_embedding(embeddings[embeddings.index_to_key])

    embeddings.add_vectors(embeddings.index_to_key, transformed_embeddings, replace=True)

    logger.info("Saving transformed embeddings to %s", os.path.join(os.path.dirname(args.embeddings_filename), "embeddings_dine.kv"))

    embeddings.save(os.path.join(os.path.dirname(args.embeddings_filename), "embeddings_dine.kv"))


if __name__ == '__main__':
    main()
