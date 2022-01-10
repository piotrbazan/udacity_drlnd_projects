from torch import nn, tensor
import torch.nn.functional as F
import torch


class FC(nn.Module):
    """
    Fully connected network
    """

    def __init__(self, input_dim, output_dim, hidden_dims):
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


class FCAC(nn.Module):
    """
    Fully connected network for A2C: similar to FC with exception there are two heads for policy and value
    """

    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.layers = nn.ModuleList()

        dims = (input_dim,) + hidden_dims
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_features, out_features))
            self.layers.append(nn.ReLU())
        self.policy_head = nn.Linear(hidden_dims[-1], output_dim)
        self.value_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.policy_head(x), self.value_head(x)


class FCQV(nn.Module):
    """
    Fully connected network for Q-value for DDPG
    """

    def __init__(self, state_dim, action_dim, hidden_dims):
        super().__init__()
        self.input_layer = nn.Linear(state_dim, hidden_dims[0])
        hidden_dims = list(hidden_dims)
        hidden_dims[0] += action_dim
        self.hidden_layers = nn.ModuleList()
        for in_features, out_features in zip(hidden_dims[:-1], hidden_dims[1:]):
            self.hidden_layers.append(nn.Linear(in_features, out_features))
        self.output_layer = nn.Linear(hidden_dims[-1], 1)

    def forward(self, state, action):
        x = F.relu(self.input_layer(state))
        x = torch.cat((x, action), dim=1).float()
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        return self.output_layer(x)


def convert_to_param(x):
    """
    Converts x to tensor (if not already a tensor) and make it a parameter
    so that .to_device() works on tensor as well
    """
    x = x if isinstance(x, torch.Tensor) else torch.tensor(x)
    x = x.float()
    return nn.Parameter(x, requires_grad=False)


class RescaleLayer(nn.Module):
    """
    Rescaling layer between values from input(min, max) to output(min, max)
    """

    def __init__(self, input_min, input_max, output_min, output_max):
        super().__init__()
        delta_input = input_max - input_min
        delta_output = output_max - output_min
        self.input_min = convert_to_param(input_min)
        self.output_min = convert_to_param(output_min)
        self.ratio = convert_to_param(delta_output / delta_input)

    def forward(self, x):
        return (x - self.input_min) * self.ratio + self.output_min


class FCDP(nn.Module):
    """
    Fully connected deterministic policy network for DDPG
    """

    def __init__(self, input_dim: int, action_bounds: tuple, hidden_dims):
        super().__init__()
        action_min, action_max = action_bounds
        net_min, net_max = torch.tanh(tensor(float('-inf'))), torch.tanh(tensor(float('inf')))

        output_dim = len(action_min)
        dims = (input_dim,) + hidden_dims + (output_dim,)
        self.layers = nn.ModuleList()
        for in_features, out_features in zip(dims[:-1], dims[1:]):
            self.layers.append(nn.Linear(in_features, out_features))
            if out_features == output_dim:
                self.layers.append(nn.Tanh())  # last layer
            else:
                self.layers.append(nn.ReLU())

        self.rescale = RescaleLayer(net_min, net_max, action_min, action_max)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.rescale(x)
