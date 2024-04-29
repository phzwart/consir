import torch
from torch import nn
import numpy as np


class SineLayer(nn.Module):
    """
    A Sine Layer as a building block for SIREN networks that uses a sinusoidal activation function.

    Attributes:
        omega_0 (float): The frequency of the sinusoidal activation function.
        is_first (bool): Indicates if this is the first layer in the network which influences the initialization.
        in_features (int): Number of input features.
        linear (nn.Linear): Linear layer used before applying the sine activation.

    Parameters:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
        is_first (bool, optional): Marks if this is the first layer in the network. Defaults to False.
        omega_0 (float, optional): Frequency parameter for the sine function. Defaults to 30.

    from https://github.com/vsitzmann/siren - under MIT license as well
    """

    def __init__(
        self, in_features, out_features, bias=True, is_first=False, omega_0=30
    ):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        """
        Initializes weights of the linear layer according to the layer's position (first or not) in the network.
        """
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(
                    -np.sqrt(6 / self.in_features) / self.omega_0,
                    np.sqrt(6 / self.in_features) / self.omega_0,
                )

    def forward(self, input):
        """
        Implements the forward pass of the layer using a sinusoidal activation function.

        Parameters:
            input (Tensor): Input tensor for the layer.

        Returns:
            Tensor: The output tensor after applying the sinusoidal activation function.
        """
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        """
        Implements the forward pass of the layer, returning both the activated output and intermediate linear layer output.

        Useful for visualization and analysis of activations.

        Parameters:
            input (Tensor): Input tensor for the layer.

        Returns:
            tuple: A tuple containing the output after sinusoidal activation and the intermediate output before activation.
        """
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    """
    A simple implementation of a SIREN (SInusoidal REpresentation Network) model.

    Attributes:
        net (nn.Sequential): The sequential container of layers which make up the SIREN model.

    Parameters:
        in_features (int): Number of input features.
        hidden_features (int): Number of features in the hidden layers.
        hidden_layers (int): Number of hidden layers in the network.
        out_features (int): Number of output features.
        outermost_linear (bool, optional): Whether the outermost layer is linear. Defaults to False.
        first_omega_0 (float, optional): Frequency parameter for the first layer's sine function. Defaults to 30.
        hidden_omega_0 (float, optional): Frequency parameter for hidden layers' sine functions. Defaults to 30.

    from https://github.com/vsitzmann/siren - under MIT license as well
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        hidden_layers,
        out_features,
        outermost_linear=False,
        first_omega_0=30,
        hidden_omega_0=30.0,
    ):
        super().__init__()

        self.net = []
        self.net.append(
            SineLayer(
                in_features, hidden_features, is_first=True, omega_0=first_omega_0
            )
        )

        for i in range(hidden_layers):
            self.net.append(
                SineLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(
                    -np.sqrt(6 / hidden_features) / hidden_omega_0,
                    np.sqrt(6 / hidden_features) / hidden_omega_0,
                )

            self.net.append(final_linear)
        else:
            self.net.append(
                SineLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        """
        Processes the input through the network and returns the output along with the input having gradients enabled.

        Parameters:
            coords (Tensor): The input coordinates tensor.

        Returns:
            tuple: The output of the network and the input tensor with gradient tracking enabled.
        """
        coords = (
            coords.clone().detach().requires_grad_(True)
        )  # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        """
        Processes the input through the network, capturing and optionally retaining gradients of intermediate activations.

        Parameters:
            coords (Tensor): The input coordinates tensor.
            retain_grad (bool, optional): If True, retains gradients for all intermediate activations. Useful for gradient-based analysis.

        Returns:
            OrderedDict: A dictionary containing all intermediate activations and outputs, keyed by their names.
        """

        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations["input"] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations[
                    "_".join((str(layer.__class__), "%d" % activation_count))
                ] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations["_".join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations
