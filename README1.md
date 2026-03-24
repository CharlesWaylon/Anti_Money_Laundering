# Initial GNN Model for AML Transaction Classification

This document summarizes the earliest graph neural network work in this project. It focuses only on the original AML transaction dataset and the first graph-based modeling pipeline used to classify suspicious transactions.

## Overview

The initial GNN effort treats AML transactions as edges in a graph of accounts. Instead of modeling each transaction as an isolated row, the project uses sender-beneficiary relationships to let the model learn from transaction structure, account behavior, and temporal patterns at the same time.

The main artifacts for this stage are:

- `GNN_Model_1.ipynb`
- `GNN_pt2.py`
- `"""GNN part 2""".py`
- `aml_syn_data`

## Dataset used

The model is built on `aml_syn_data`, the main transaction dataset in the project.

Based on the saved files in this repository, the dataset contains:

- 1,484,536 rows
- 13 columns
- labels `GOOD` and `BAD`
- class balance of about 79.05% `GOOD` and 20.95% `BAD`

This means the task is an imbalanced binary classification problem, which matters when choosing metrics and loss weighting.

## Modeling idea

The GNN setup reframes the data as a transaction graph:

- nodes represent accounts
- edges represent transactions from sender account to beneficiary account
- edge labels represent whether the transaction is `GOOD` or `BAD`

This approach is useful for AML because suspicious behavior often depends on network structure, repeated counterparties, account roles, and relational patterns that flat row-wise models can miss.

## Feature engineering

The initial pipeline in `GNN_Model_1.ipynb` and `GNN_pt2.py` builds both edge features and node features.

### Edge features

The saved code shows the following transaction-level features were used:

- log-transformed `USD_amount`
- `hour`
- `dayofweek`
- `month`
- one-hot encoded `Transaction_Type`

Continuous edge features were standardized using statistics fit on the training split only.

### Node features

The follow-on GNN script builds per-account node features from the training data, including:

- sent transaction count
- total sent amount
- average sent amount
- received transaction count
- total received amount
- average received amount
- dominant sender country per account, one-hot encoded

These node features let the model combine account-level behavior with per-transaction attributes.

## Data splitting

The transaction records were sorted by `Time_step` and split in chronological order:

- 70% train
- 15% validation
- 15% test

This is a sensible choice for AML-style data because it reduces temporal leakage compared with random splitting.

## Initial GNN architecture

The main graph model defined in `GNN_pt2.py` is an edge-classification GraphSAGE network:

- two `SAGEConv` layers produce node embeddings
- for each transaction edge, the model concatenates:
  - source node embedding
  - destination node embedding
  - edge attributes
- a small MLP outputs a binary logit for whether the transaction is suspicious

Training uses:

- `BCEWithLogitsLoss`
- positive-class weighting based on class imbalance
- Adam optimizer
- 50 epochs
- learning rate `1e-4`
- weight decay `1e-4`

## Saved training progress

`GNN_Model_1.ipynb` preserves a training log for the first model run.

The recorded validation F1:

- started around `0.4608`
- dipped in the middle of training
- improved later in training
- reached about `0.6220` by epoch 50

This suggests the graph model was learning useful signal, even if the experiment was still in an exploratory stage.

## Additional baselines and follow-up experiments

The notebook also includes comparisons and extensions beyond the first GraphSAGE run:

- logistic regression baselines on edge features
- decision tree baselines
- trivial baselines such as predicting all `GOOD`
- label-permutation sanity checking
- a Graph Attention Network (`GAT`) variant
- simple crosstab analysis linking label rates to transaction type, amount bins, and country combinations

That makes the notebook more than a single model run. It acts as an early research workspace for understanding what information is predictive and whether graph structure adds value beyond standard tabular features.

## What this stage accomplished

The initial GNN work established a reusable graph-learning direction for the project:

- it converted the AML dataset into a node-edge representation
- it built a first edge-classification GNN for suspicious transaction detection
- it added temporal splitting and imbalance-aware training
- it recorded an improving validation F1 curve, with a best saved value around `0.62`
- it set up comparisons against non-graph baselines

## Limitations of the current GNN stage

From the saved files, this work still looks like an initial research prototype rather than a finalized modeling pipeline. A few likely gaps remain:

- the notebook mixes feature engineering, modeling, baselines, and exploratory analysis in one place
- final test results are not documented as clearly as the validation trajectory
- experiment settings and conclusions are not yet packaged into a reproducible report
- the relationship between `GNN_Model_1.ipynb` and the follow-on scripts could be documented more clearly

## Suggested next steps

- cleanly separate preprocessing, graph construction, training, and evaluation into scripts
- save final validation and test metrics in a dedicated results file
- compare GraphSAGE and GAT under the same evaluation setup
- add precision-recall metrics and class-specific recall for the `BAD` class
- document the exact graph schema and feature definitions used for each run

## Summary

The initial GNN stage of this project explored AML detection as an edge-classification problem on a transaction graph. Transactions were modeled as edges between accounts, enriched with temporal and transaction-type features, while nodes captured account-level behavior. The first saved training run shows validation F1 improving to roughly `0.62`, which makes this a promising early graph-based baseline for the broader AML modeling work.
