# NetAudit: Interpretable Population-scale Node Embeddings

This repository contains code to create node embeddings for the Dutch population network created by Statistics Netherlands. The population network data are not public but can be accessed under certain [conditons](\url{https://www.cbs.nl/nl-nl/onze-diensten/maatwerk-en-microdata/microdata-zelf-onderzoek-doen}) through Statistics Netherlands.

While the code for creating the embeddings is generic, we also include code for predicting three outcomes from embeddings and/or covariates using machine learning models. This code requires linking data from the LISS panel with microdata from Statistic Netherlands. The LISS panel data can be accessed through the [LISS Archive](https://www.lissdata.nl/how-it-works-archive). For details see this paper (LINK FORTHCOMING).

The code in this repository is written to run in the remote access environment by Statistics Netherlands. This environment has certain computational constraints that limits the use of existing Python packages on which this code depends. As of February 2025, this code made it possible to create node embeddings for the entire Dutch population using only the ressources in the remote access environment.

All scripts have help texts that can be inspected with:

```
python scripts/example.py --help
```

## Creating Population-Scale Node Embeddings

To create node embeddings run these scripts in order:

1. `create_edgelist_from_csv.py`: Creates an combined edgelist from CSV files containing edgelists for different layers of the population network
2. `create_adjacency_list_batched.py`: Processes batches of the combined edgelist and creates an adjacency list for each batch
3. `merge_adjacency_batches.py`: Merges adjacency list batches into a single adjacency list
4. Create embeddings with one of two methods:
    1. `train_deepwalk.py`: Train DeepWalk embeddings from an adjacency list
    2. `train_line.py`: Train LINE embeddings from an edgelist and an adjacency list

Creating the adjacency list in batches was necessary given the large size of the population network and the memory constraints in the environment.

Example with made-up test network:
```
# Create edgelist
python scripts/create_edgelist_from_csv.py --csv-string data/raw/test_network_*.csv --edgelist-filename data/processed/test_edgelist.parquet

# Create adjacency list
python scripts/create_adjacency_list_batched.py --edgelist-filename data/processed/test_edgelist.parquet --outdir data/processed/test_adjacency_batches

python scripts/merge_adjacency_batches.py --adjacency-dir data/processed/test_adjacency_batches/ --adjacency-filename data/processed/test_adjacency_list.parquet

# Train DeepWalk embeddings
python scripts/train_deepwalk.py --adjacency-filename data/processed/test_adjacency_list.parquet
```

## Disentangling Embedding Dimensions

This code adapts an exising implementation to apply the [DINE](http://arxiv.org/abs/2310.01162) framework to the embeddings.

To make the embedding dimensions more interpretable, run:

5. `train_dine.py`: Train a regularizing auto-encoder as part of the DINE framework on embeddings to distentangle embedding dimensions

In comparison to training the embeddings, training the auto-encoder should go rather fast (at least an order of magnitude faster).

To calculate edge utilities, run:

6. `calc_edge_utilities.py`: Calculate edge utilities from an adjacency list and embeddings for a specific embedding dimension

## Predicting Outcomes

To train a machine learning model to predict an outcome from embeddings and/or covariates, run:

7. `train_prediction_model.py`: Train a machine learning prediction model 

This code requires a preprocessed file with covariates and the outcome variables. See the Supplementary Material of this paper (LINK FORTHCOMING) for details.

## Community Detection

We also provide code to perform community detection with the Louvain algorithm on the population network:

8. `detect_communties.py`: Detect communties from an adjacency list using the Louvain algorithm

## Requirements

All Python packages needed to run the code are in `requirements.txt`. The code is intended to run on Windows and Linux.

## License

The code is licensed under the Apache 2.0 License. This means that it can be used, modified and redistributed for free, even for commercial purposes.

## Contact

Malte Lüken (m.luken@esciencecenter.nl)
