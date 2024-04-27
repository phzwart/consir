import torch
from torch import nn
import numpy as np
import torch
from torch import nn
import numpy as np


class SirenLayer(nn.Module):
    """
    A Sine Layer for a SIREN network, using a sinusoidal activation function.

    Attributes:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        w0 (float): The frequency of the sinusoidal activation. Defaults to 1.0.
    """

    def __init__(self, in_features, out_features, w0=1.0):
        """
        Initializes the SirenLayer with necessary layers and parameters.

        Args:
            in_features (int): Number of features in the input tensor.
            out_features (int): Number of features in the output tensor.
            w0 (float): Frequency parameter for the sinusoidal activation.
        """
        super().__init__()
        self.w0 = w0
        self.linear = nn.Linear(in_features, out_features)
        # Initialization with uniform distribution considering w0
        with torch.no_grad():
            self.linear.weight.uniform_(
                -np.sqrt(6 / in_features) / w0, np.sqrt(6 / in_features) / w0
            )

    def forward(self, x):
        """
        Forward pass of the SirenLayer.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor after applying sinusoidal activation.
        """
        return torch.sin(self.w0 * self.linear(x))


class SirenModel(nn.Module):
    """
    A SIREN (SInusoidal REpresentation Networks) model that utilizes layers of
    sine functions as activations to process inputs, useful for tasks that benefit
    from periodic activation functions like image and audio processing.

    Attributes:
        input_features (int): Number of input features.
        output_features (int): Number of output features.
        hidden_layers (list of int): Sizes of hidden layers.
        w0 (float): Frequency parameter for the sinusoidal activation across all layers.
    """

    def __init__(
        self, input_features, output_features, hidden_layers=[32, 32, 32], w0=1
    ):
        """
        Initializes the SirenModel with a sequence of SirenLayers.

        Args:
            input_features (int): Number of features in the input tensor.
            output_features (int): Number of features in the output tensor.
            hidden_layers (list of int): List specifying the number of neurons in each hidden layer.
            w0 (float): Frequency parameter for the sinusoidal activation for all layers.
        """
        super().__init__()
        self.input_features = input_features
        self.output_features = output_features
        layers = []
        in_features = input_features  # Customize the input features
        for out_features in hidden_layers:
            layers.append(SirenLayer(in_features, out_features, w0=w0))
            in_features = out_features
        layers.append(
            SirenLayer(in_features, output_features, w0=w0)
        )  # Customize the output features
        self.net = nn.Sequential(*layers)

    def forward(self, coords):
        """
        Forward pass of the SirenModel.

        Args:
            coords (Tensor): Input tensor containing the coordinates or features.

        Returns:
            Tensor: Output tensor from the final layer of the model.
        """
        return self.net(coords)
