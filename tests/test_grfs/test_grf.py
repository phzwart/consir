import pytest
import torch
from consir.functions.grf.generator import generate_grf
from consir.functions.grf.transformer import transform_grf


@pytest.mark.parametrize("alpha", [0.1, 0.2, 0.5])
def test_generate_grf_shape_and_type(alpha):
    """
    Test GRF generation for different alpha values.
    """
    grid_size = 100
    grf = generate_grf(grid_size, alpha)
    assert grf.shape == (grid_size, grid_size), "GRF shape is incorrect"
    assert torch.is_tensor(grf), "GRF should be a torch tensor"


@pytest.mark.parametrize("alpha, quantile", [(0.1, 0.25), (0.2, 0.5), (0.5, 0.75)])
def test_transform_grf_changes(alpha, quantile):
    """
    Test GRF generation for different quantiles.
    """
    grid_size = 100
    kappa = 10
    grf = generate_grf(grid_size, alpha)
    transformed_grf = transform_grf(grf, kappa, quantile)
    assert torch.is_tensor(transformed_grf), "Transformed GRF should be a torch tensor"
    # this is a simplistic test; we would probably want to add checks on the values themselves to ensure correct transformation


def test_transform_grf_values():
    grid_size = 100
    alpha = 0.2
    kappa = 10
    quantile = 0.5

    grf = generate_grf(grid_size, alpha)
    original_grf_mean = grf.mean()

    transformed_grf = transform_grf(grf, kappa, quantile)
    transformed_grf_mean = transformed_grf.mean()

    assert (
        transformed_grf_mean != original_grf_mean
    ), "Transformation should change the mean of the GRF"
