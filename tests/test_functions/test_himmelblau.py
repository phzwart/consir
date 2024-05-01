import torch
import pytest
from consir.functions.himmelblau import (
    himmelblau_function,
)  # Adjust this import to your file structure


def test_himmelblau_correctness():
    # Known points and their expected Himmelblau function values
    test_points = torch.tensor(
        [
            [3.0, 2.0],  # Known minimum
            [-2.805118, 3.131312],  # Another known minimum
            [-3.779310, -3.283186],  # Another known minimum
            [3.584428, -1.848126],  # Another known minimum
        ]
    )
    expected_values = torch.tensor([0, 0, 0, 0], dtype=torch.float32)

    # Calculate Himmelblau values
    calculated_values = himmelblau_function(test_points)

    # Assert all calculated values are close to expected values
    assert torch.allclose(
        calculated_values, expected_values, atol=1e-5
    ), "Function values are incorrect at known minima."


def test_himmelblau_input_validation():
    # Test with incorrect input dimensions
    with pytest.raises(IndexError):
        wrong_dim_input = torch.tensor([1.0, 2.0])  # This should be 2D (n, 2)
        himmelblau_function(wrong_dim_input)

    # Test with correct dimensions but wrong shape
    with pytest.raises(IndexError):
        wrong_shape_input = torch.tensor(
            [[1.0], [2.0]]
        )  # Correct 2D shape but incorrect inner dimension
        himmelblau_function(wrong_shape_input)


def test_himmelblau_edge_cases():
    # Test with large values
    large_values = torch.tensor([[1e10, 1e10], [-1e10, -1e10]])
    large_value_outputs = himmelblau_function(large_values)
    assert (
        large_value_outputs.all() > 0
    ), "Function values should be large and positive for large inputs."

    # Test with zero input
    zero_input = torch.tensor([[0.0, 0.0]])
    zero_output = himmelblau_function(zero_input)
    assert zero_output.item() == 170, "Function value should be exactly 170 at (0,0)."


# Additional tests can be added here for more comprehensive coverage
