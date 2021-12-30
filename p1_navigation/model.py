import torch
from torch import nn


class DqnModel(nn.Module):

    def __init__(self, input_dim, output_dim, hidden_dims, seed=42):
        super().__init__()
        self.layers = nn.ModuleList()

        dims = (input_dim,) + hidden_dims + (output_dim,)
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_features, out_features))
            if out_features != output_dim:  # if not last layer
                self.layers.append(nn.ReLU())

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
