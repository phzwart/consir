"""Test for SIREN functions"""

import pytest
import torch
import numpy as np
from torch import nn

# Assuming the new Siren classes are in a module named my_siren_module
from consir.sirens.siren import SineLayer, Siren


def test_sine_layer_initialization():
    """
    Test the initialization of the SineLayer to ensure weights are set correctly.
    """
    in_features = 5
    out_features = 10
    is_first = False
    omega_0 = 30.0
    layer = SineLayer(in_features, out_features, is_first=is_first, omega_0=omega_0)

    # Check if weights are within the expected range for not the first layer
    expected_lower_bound = -np.sqrt(6 / in_features) / omega_0
    expected_upper_bound = np.sqrt(6 / in_features) / omega_0
    assert torch.all(layer.linear.weight <= expected_upper_bound) and torch.all(
        layer.linear.weight >= expected_lower_bound
    ), "Weights are not initialized within the expected range."


def test_sine_layer_forward():
    """
    Test the forward pass of the SineLayer.
    """
    in_features = 5
    out_features = 10
    layer = SineLayer(in_features, out_features)
    input_tensor = torch.randn(1, in_features)
    output = layer(input_tensor)

    # Check the output shape and type
    assert output.shape == (1, out_features), "Output shape is incorrect."
    assert torch.is_tensor(output), "Output is not a tensor."


def test_siren_initialization():
    """
    Test the initialization of the Siren model to verify correct layer setup.
    """
    in_features = 2
    hidden_features = 32
    hidden_layers = 3
    out_features = 1
    model = Siren(in_features, hidden_features, hidden_layers, out_features)

    # Check the number of layers
    expected_num_layers = hidden_layers + 2  # Including first and last layers
    assert (
        len(model.net) == expected_num_layers
    ), "Incorrect number of layers in the model."


def test_siren_forward():
    """
    Test the forward pass of the Siren model.
    """
    in_features = 2
    hidden_features = 32
    hidden_layers = 3
    out_features = 1
    model = Siren(in_features, hidden_features, hidden_layers, out_features)
    input_tensor = torch.randn(10, in_features)  # Batch of 10
    output, _ = model(input_tensor)

    # Check output shape and type
    assert output.shape == (10, out_features), "Output shape is incorrect."
    assert torch.is_tensor(output), "Output is not a tensor."
