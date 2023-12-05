import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Iterable


class TransformerLayer(nn.Module):
    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            ff_dim: int,
            dropout_rate: float=0.1,
        ):
        super(TransformerLayer, self).__init__()

        self.att = nn.MultiheadAttention(num_heads=num_heads, embed_dim=embed_dim, dropout=dropout_rate)
        self.ffn = nn.Sequential(
            *[nn.Linear(embed_dim, ff_dim), nn.ReLU(), nn.Linear(ff_dim, embed_dim),]
        )

        self.att_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ff_layernorm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.ff_dropout = nn.Dropout(dropout_rate)

    def forward(self, inputs):
        attn_output, _ = self.att(inputs, inputs, inputs)
        out1 = self.ff_layernorm(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.ff_dropout(ffn_output)
        return self.ff_layernorm(out1 + ffn_output)


class TabTransformer(pl.LightningModule):

    def __init__(
            self,
            dataset,
            dim_out: int,
            embed_dim: int = 16,
            ff_dim: int = 16,
            depth: int = 5,
            heads: int = 4,
            mlp_hidden: Iterable = {"linear1": 16, "activation1": "ReLU", "linear2": 16, "activation2": "ReLU"},
            lr: float=1e-3,
            batch_size: int=20,
        ):
        
        super(TabTransformer, self).__init__()

        # embeddings
        self.embedding_layers = nn.ParameterList([])
        class_sizes = dataset.nu
        for number_of_classes in class_sizes:
            self.embedding_layers.append(nn.Embedding(number_of_classes, embed_dim))

        # transformers
        transformers = []
        for _ in range(depth):
            transformers.append(TransformerLayer(embed_dim, heads, ff_dim))
        self.transformers = nn.Sequential(*transformers)
        self.flatten_transformer_output = nn.Flatten()

        # mlp layers
        mlp_layers = []
        current_dim = embed_dim * len(class_sizes)
        for layer_type, value in mlp_hidden.items():
            if layer_type[:6] == "linear":
                mlp_layers.append(nn.Linear(current_dim, value))
                current_dim = value
            else:
                try:
                    activation_fn = getattr(nn, value)
                except AttributeError:
                    raise ValueError(f"Activation function '{value}' not found in torch.nn module.")
                mlp_layers.append(activation_fn())

        mlp_layers.append(nn.Linear(current_dim, dim_out))

        self.mlp = nn.Sequential(*mlp_layers)

        self.loss = nn.L1Loss()

        self.lr = lr
        self.batch_size = batch_size

        self.train_dataloader_ = dataset.get_dataloader()
        self.valid_dataloader_ = dataset.get_dataloader(valid=True)

    def forward(self, inputs):
        embeddings = []
        for i in range(len(inputs[0])):
            embeddings.append(self.embedding_layers[i](inputs[:, i]))
        transformer_inputs = torch.stack(embeddings, axis=1)

        # transformers
        
        mlp_inputs = torch.flatten(self.transformers(transformer_inputs), start_dim=1)

        return self.mlp(mlp_inputs)

    def regularization(self):
        return 1e-3 * sum([(p ** 2).sum() for p in self.parameters()])

    def training_step(self, batch, _):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        reg = self.regularization()
        self.log("train_loss", loss)
        self.log("reg", reg)
        return loss

    def validation_step(self, batch, _):
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)

    def train_dataloader(self):
        return self.train_dataloader_

    def val_dataloader(self):
        return self.valid_dataloader_
