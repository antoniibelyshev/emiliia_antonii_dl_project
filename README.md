# Single-Cell Perturbations

## Introduction

In our project we aim to tackle a scientific problem of accurate prediction of the chemical perturbations in new cell types having known very little about perturbations in those cell types but having known lots of perturbations for another cell types. The initial dataset that we are working with contains information about measured chemical perturbations for some set of cells as well as information about the cell type, compound type and chemical structure, and weather the patient is in the control group. For each entry the differential expression of the perturbation is known. Numerically the differential expression is represented as a sequence of ~18000 numbers.

## Data preprocessing

We transform the target sequences using truncated svd to reduce the dimensionality of the target vector. Then, after the prediction of the svg representation, we have to perform an (pseudo) inverse transform.

The given features are categorical variables, so we encode them into the nonnegative whole numbers.

## Model architecture

We decided to use tab transformer (https://arxiv.org/pdf/2012.06678.pdf) consisting of the following parts:

- A column embedding, one embedding vector for each categorical feature, is added (point-wise) to the categorical feature embedding.
- The embedded categorical features are fed into a stack of Transformer blocks. Each Transformer block consists of a multi-head self-attention layer followed by a feed-forward layer.
- The outputs of the final Transformer layer, which are the contextual embeddings of the categorical features, are fed into a final MLP block.

## Experiments

We provide a few experiments with various hyperparameters. The experiments could be performed by running the script run_experiments.sh. The experiments logs are collected in the weights and biases, so you should provide your key in configs/meta.yaml.

You can run experiments with different parameters by setting these parameters in the config files (as shown in the examples). Generally, you have to create one file for each part: dataset, model, and trainer, as well as combine them with a single file in /configs and modify meta.yaml file. You also should specify the name of config in experiments.py file.

If you want to perform one experiment for some set of hyperparameters, then set type = "single" in config/meta.yaml. If you want to perform multiple experiments with various values of hyperparameters, you should provide lists of values for parameters that you want to variate. You also should set type = "simultanious" or type = "all" in config/meta.yaml. "simultanious" means that the iteration over the possible values of the hyperparameters will be simultanious (and therefore the provided lists whould have the same length). "all" means that all possible sets of values will be checked.
