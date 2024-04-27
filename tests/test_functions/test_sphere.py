import pytest
import torch

from consir.functions.sphere import (
    sphere_function,
)


@pytest.mark.parametrize(
    "inputs, expected, tol",
    [
        (
            torch.zeros(1, 10).float(),
            torch.tensor([0.0]),
            0.001,
        ),  # Expect zero input to have function value close to 0 (Sphere minimum)
        (
            torch.ones(1, 10).float(),
            torch.tensor([10.0]),
            0.001,
        ),  # Test with all ones, we expect 10.0
        (
            torch.full((1, 10), 2),
            torch.tensor([40.0]),
            0.001,
        ),  # Test with all twos, we expect 40.0
    ],
)
def test_sphere(inputs, expected, tol):
    """
    Test the sphere_function with various inputs and check if the output matches the expected values.
    """
    result = sphere_function(inputs.float())
    print(inputs, expected, result)
    torch.testing.assert_close(
        result, expected, rtol=tol, atol=tol
    )  # Using a tolerance of 0.001 for floating point comparison


@pytest.mark.parametrize("dim", [1, 5, 10, 100])
def test_sphere_high_dimensionality(dim):
    """
    Test the sphere_function with increasing dimensionality to ensure it handles high dimensions.
    """
    input_tensor = torch.randn(1, dim)
    # Since it's hard to know the exact output, we test if the function runs without errors
    assert sphere_function(input_tensor) is not None


def test_sphere_zero_vector():
    """
    Specifically test the Sphere function at the global minimum, which should be near zero.
    """
    zero_vector = torch.zeros(1, 10).type(torch.FloatTensor)
    expected_value = torch.tensor([0.0])
    result = sphere_function(zero_vector)
    torch.testing.assert_close(result, expected_value, atol=0.001, rtol=0.001)
