import pytest
import torch

from consir.functions.ackley import (
    ackley_function,
)


@pytest.mark.parametrize(
    "inputs, expected, tol",
    [
        (
            torch.zeros(1, 10).float(),
            torch.tensor([0.0]),
            0.001,
        ),  # Expect zero input to have function value close to 0 (Ackley minimum)
        (torch.ones(1, 10).float(), torch.tensor([3.625]), 0.001),  # Test with all ones
        (torch.full((1, 10), 2), torch.tensor([6.594]), 0.001),  # Test with all twos
    ],
)
def test_ackley(inputs, expected, tol):
    """
    Test the ackley_function with various inputs and check if the output matches the expected values.
    """
    result = ackley_function(inputs.float())
    print(inputs, expected, result)
    torch.testing.assert_close(
        result, expected, rtol=tol, atol=tol
    )  # Using a tolerance of 0.001 for floating point comparison


@pytest.mark.parametrize("dim", [1, 5, 10, 100])
def test_ackley_high_dimensionality(dim):
    """
    Test the ackley_function with increasing dimensionality to ensure it handles high dimensions.
    """
    input_tensor = torch.randn(1, dim)
    # Since it's hard to know the exact output, we test if the function runs without errors
    assert ackley_function(input_tensor) is not None


def test_ackley_zero_vector():
    """
    Specifically test the Ackley function at the global minimum, which should be near zero.
    """
    zero_vector = torch.zeros(1, 10).type(torch.FloatTensor)
    expected_value = torch.tensor([0.0])
    result = ackley_function(zero_vector)
    torch.testing.assert_close(result, expected_value, atol=0.001, rtol=0.001)
