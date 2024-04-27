import pytest
import torch

from consir.functions.rosenbrock import (
    rosenbrock_function,
)


@pytest.mark.parametrize(
    "inputs, expected, tol",
    [
        (
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([0.0]),
            0.001,
        ),  # Global minimum @ (1,1)
        (
            torch.tensor([[1.0, 1.0, 1.0]]),
            torch.tensor([0.0]),
            0.001,
        ),  # Global minimum in 3D
    ],
)
def test_rosenbrock(inputs, expected, tol):
    """
    Test the rosenbrock_function with various inputs and check if the output matches the expected values.
    """
    result = rosenbrock_function(inputs.float())
    torch.testing.assert_close(result, expected, rtol=tol, atol=tol)


@pytest.mark.parametrize("dim", [2, 3, 5, 10])
def test_rosenbrock_high_dimensionality(dim):
    """
    Test the rosenbrock_function with increasing dimensionality to ensure it handles high dimensions.
    """
    # Generate a tensor close to the Rosenbrock minimum (1,1,...,1)
    input_tensor = torch.full((1, dim), 1.0) + torch.randn(1, dim) * 0.1
    result = rosenbrock_function(input_tensor)
    assert result is not None


def test_rosenbrock_zero_vector():
    """
    Test the Rosenbrock function far from the global minimum, which should not be zero.
    """
    zero_vector = torch.zeros(1, 10).type(torch.FloatTensor)
    expected_value = torch.tensor(
        [9.0]
    )  # Because the formula results in (10-1)(100 * (0-0^2)^2 + (1-0)^2) for each pair
    result = rosenbrock_function(zero_vector)
    torch.testing.assert_close(result, expected_value, atol=0.001, rtol=0.001)
