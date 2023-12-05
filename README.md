# Single-Cell Perturbations

## Introduction

In our project we aim to tackle a scientific problem of accurate prediction of the chemical perturbations in new cell types having known very little about perturbations in those cell types but having known lots of perturbations for another cell types. The initial dataset that we are working with contains information about measured chemical perturbations for some set of cells as well as information about the cell type, compound type and chemical structure, and weather the patient is in the control group. For each entry the differential expression of the perturbation is known. Numerically the differential expression is represented as a sequence of ~18000 numbers.

## Data preprocessing

We transform the target sequences using truncated svd to reduce the dimensionality of the target vector. Then, after the prediction of the svg representation, we have to perform an (pseudo) inverse transform.

The given features are categorical variables, so we encode them into the nonnegative whole numbers.

## Model architecture

We have decided to implement a tab transformer architecture, as detailed in the paper [TabTransformer: Tabular Data Modeling Using Contextual Embeddings](https://arxiv.org/pdf/2012.06678.pdf). This model is composed of the following key components:

1. **Column Embedding for Categorical Features:**
   - **Purpose:**
     - To handle categorical features effectively, the model employs a column embedding strategy.
     - Each categorical feature is associated with an embedding vector.
   - **Explanation:**
     - Categorical features, being non-numeric, are transformed into continuous vector representations through embeddings.
     - The pointwise addition of the column embedding to the categorical feature embedding enhances the model's ability to capture nuanced relationships within the categorical feature space.
   - **Significance:**
     - Enables the model to learn complex interactions between different categorical features.
     - Enhances the expressiveness of the model by incorporating categorical context into the embeddings.

2. **Transformer Blocks for Categorical Features:**
   - **Purpose:**
     - To capture long-range dependencies and intricate patterns within the embedded categorical features.
   - **Explanation:**
     - The embedded categorical features are processed through a stack of Transformer blocks.
     - Each Transformer block comprises a multi-head self-attention layer, allowing the model to weigh the importance of different feature interactions differently.
     - Followed by a feed-forward layer for non-linear transformations.
   - **Significance:**
     - The multi-head self-attention mechanism enables the model to focus on relevant feature dependencies, facilitating more effective learning of complex relationships.
     - Transformer blocks provide a flexible and scalable architecture for capturing diverse patterns in tabular data.

3. **Final MLP Block for Contextual Embeddings:**
   - **Purpose:**
     - To derive contextual embeddings of the categorical features from the outputs of the final Transformer layer.
   - **Explanation:**
     - The contextual embeddings obtained after processing through the Transformer blocks are fed into a final Multi-Layer Perceptron (MLP) block.
     - The MLP block is responsible for further non-linear transformations and abstraction of the contextual embeddings.
   - **Significance:**
     - Facilitates the extraction of high-level representations that capture the context and relationships learned by the model.
     - Provides a compact and informative representation of the input tabular data, suitable for downstream tasks such as classification or regression.

## Experiments

We provide various experiments where we try to use wide range of hyperparameters in pur model, data preprocessing and training process. The experiments could be reproduced by specifying the config_name in the experiments.py and running the script run_experiments.sh.

The experiments logs are collected in the weights and biases, so you should provide your key in configs/meta.yaml. After the experiment you can find all the information about it in the wandb logs: values of the hyperparameters, metrics, type of the model, etc.

## New experiments

In order to research, develop and use our algorithm conveniently, we created well-structured configuration files, where we specify the configuration for each experiment.

The configs are structured as follows:
  - For each experiment, there is a file configs/name.yaml. It specifies which configs should be used for the components: dataset, model, trainer. It also loads the metainformation about the experiment from the meta.yaml, including wandb ssh key, experiment name, experiment type, etc.
  - In the directories configs/dataset/, configs/model and configs/trainer we store the configs for corresponding subsets of parameters.
  - The specific config name that is used for current run is specified in the decorator for the main() in experiments.py

If you would like to perform one of existing experiments, you should specify its name in the experiments.py and modify some parameters in configs/meta.yaml.

To run a new experiment with another hyperparameters, you should create an additional .yaml file in configs/, where you specify the configs for the components. You also probably should create new config files for the components, however it is also possible to use the configuration for some components used in different experiments. When creating a new experiment setup, you should keep the structure of the config files as in the example experiments.

