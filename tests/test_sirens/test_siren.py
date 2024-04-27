"""Test for siner functions"""

import pytest
import torch
from consir.sirens import siren


def test_siren_layer_initialization():
    """
    Test the initialization of the SirenLayer to ensure weights are set correctly.
    """
    in_features = 5
    out_features = 10
    w0 = 1.0
    layer = siren.SirenLayer(in_features, out_features, w0)

    # Check if weights are within the expected range
    expected_lower_bound = -torch.sqrt(torch.tensor(6.0 / in_features)) / w0
    expected_upper_bound = torch.sqrt(torch.tensor(6.0 / in_features)) / w0
    assert torch.all(layer.linear.weight <= expected_upper_bound) and torch.all(
        layer.linear.weight >= expected_lower_bound
    ), "Weights are not initialized within the expected range."


def test_siren_layer_forward():
    """
    Test the forward pass of the SirenLayer.
    """
    in_features = 5
    out_features = 10
    w0 = 1.0
    layer = siren.SirenLayer(in_features, out_features, w0)
    input_tensor = torch.randn(1, in_features)
    output = layer(input_tensor)

    # Check the output shape and type
    assert output.shape == (1, out_features), "Output shape is incorrect."
    assert torch.is_tensor(output), "Output is not a tensor."


def test_siren_model_initialization():
    """
    Test the initialization of the SirenModel to verify correct layer setup.
    """
    input_features = 2
    output_features = 1
    hidden_layers = [32, 32, 32]
    model = siren.SirenModel(input_features, output_features, hidden_layers)

    # Check the number of layers
    expected_num_layers = len(hidden_layers) + 1  # Including the output layer
    assert (
        len(model.net) == expected_num_layers
    ), "Incorrect number of layers in the model."


def test_siren_model_forward():
    """
    Test the forward pass of the SirenModel.
    """
    input_features = 2
    output_features = 1
    hidden_layers = [32, 32, 32]
    model = siren.SirenModel(input_features, output_features, hidden_layers)
    input_tensor = torch.randn(10, input_features)  # Batch of 10
    output = model(input_tensor)

    # Check output shape and type
    assert output.shape == (10, output_features), "Output shape is incorrect."
    assert torch.is_tensor(output), "Output is not a tensor."
