# NetAudit: Interpretable Population-scale Node Embeddings

This repository contains code to create node embeddings for the Dutch population network created by Statistics Netherlands.

To create node embeddings run these scripts in order:

1. `create_edgelist_from_csv.py`: Creates an combined edgelist from CSV files containing edgelists for different layers of the population network
2. `create_adjacency_list_batched.py`: Processes batches of the combined edgelist and creates an adjacency list for each batch
3. `merge_adjacency_batches.py`: Merges adjacency list batches into a single adjacency list
4. Create embeddings with one of two methods:
    1. `train_deepwalk.py`: Train DeepWalk embeddings from an adjacency list
    2. `train_line.py`: Train LINE embeddings from an edgelist and an adjacency list

To make the embedding dimensions more interpretable, run:

5. `train_dine.py`: Train a regularizing auto-encoder as part of the DINE framework on embeddings to distentangle embedding dimensions

To train a machine learning model to predict an outcome from embeddings and/or covariates, run:

6. `train_prediction_model.py`: Train a machine learning prediction model 
