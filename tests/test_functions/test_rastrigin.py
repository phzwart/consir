import pytest
import torch

from consir.functions.rastrigin import (
    rastrigin_function,
)


@pytest.mark.parametrize(
    "inputs, expected, tol",
    [
        (
            torch.zeros(1, 10).float(),
            torch.tensor([0.0]),
            0.001,
        ),  # Global minimum @ zero
        (
            torch.full((1, 10), 2 * torch.pi),
            torch.tensor([515.48]),
            0.001,
        ),  # Local minimum @ 2*pi in each dimension. Note: This is outside the typical search domain for the rastrigin function,
        # which, because of the periodic nature of cosine, should not be an issue and can help us better understand the function across a more robust domain. Remove if neccessary.
        (torch.ones(1, 10).float(), torch.tensor([10.0]), 0.001),  # Test with all ones
        (torch.full((1, 10), 2), torch.tensor([40.0]), 0.001),  # Test with all twos
    ],
)
def test_rastrigin(inputs, expected, tol):
    """
    Test the rastrigin_function with various inputs and check if the output matches the expected values.
    """
    result = rastrigin_function(inputs.float())
    print(inputs, expected, result)
    torch.testing.assert_close(
        result, expected, rtol=tol, atol=tol
    )  # Using a tolerance of 0.001 for floating point comparison


@pytest.mark.parametrize("dim", [1, 5, 10, 100])
def test_rastrigin_high_dimensionality(dim):
    """
    Test the rastrigin_function with increasing dimensionality to ensure it handles high dimensions.
    """
    input_tensor = torch.randn(1, dim)
    # Since it's hard to know the exact output, we test if the function runs without errors
    assert rastrigin_function(input_tensor) is not None


def test_rastrigin_zero_vector():
    """
    Specifically test the rastrigin function at the global minimum, which should be near zero.
    """
    zero_vector = torch.zeros(1, 10).type(torch.FloatTensor)
    expected_value = torch.tensor([0.0])
    result = rastrigin_function(zero_vector)
    torch.testing.assert_close(result, expected_value, atol=0.001, rtol=0.001)
