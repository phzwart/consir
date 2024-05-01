import torch
import pytest
from consir.functions.grf.grf_interpolator import GridInterpolator
from consir.functions.himmelblau import himmelblau_function


# Setup a fixture for the GridInterpolator
@pytest.fixture
def interpolator():
    grid = torch.tensor(
        [[i + j for j in range(5)] for i in range(5)], dtype=torch.float32
    )
    domain = [(0, 4), (0, 4)]
    return GridInterpolator(grid, domain)


def test_interpolation_at_corners(interpolator):
    points = torch.tensor(
        [
            [0, 0],  # Min x, min y
            [4, 0],  # Max x, min y
            [0, 4],  # Min x, max y
            [4, 4],  # Max x, max y
        ],
        dtype=torch.float32,
    )
    expected_values = torch.tensor([[0], [4], [4], [8]], dtype=torch.float32)
    interpolated_values = interpolator(points)
    assert torch.allclose(
        interpolated_values, expected_values
    ), "Interpolation at corners failed"


def test_interpolation_at_center(interpolator):
    points = torch.tensor([[2, 2]], dtype=torch.float32)
    expected_values = torch.tensor([[4]], dtype=torch.float32)
    interpolated_values = interpolator(points)
    assert torch.allclose(
        interpolated_values, expected_values
    ), "Interpolation at center failed"


def test_interpolation_out_of_bounds(interpolator):
    points = torch.tensor([[5, 5]], dtype=torch.float32)
    expected_values = torch.tensor([[0]], dtype=torch.float32)
    interpolated_values = interpolator(points)
    assert torch.allclose(
        interpolated_values, expected_values
    ), "Out of bounds interpolation should return 0"


def test_interpolator_views():
    N = 10
    factor = 4

    x = torch.linspace(-4, 4, N)
    coords_1 = torch.cartesian_prod(x, x)
    f_1 = himmelblau_function(coords_1)
    f_1_square = f_1.view(N, N)

    x = torch.linspace(-4, 4, N * factor)
    coords_2 = torch.cartesian_prod(x, x)
    f_2 = himmelblau_function(coords_2)

    interpolator = GridInterpolator(f_1_square, domain=[(-4, 4), (-4, 4)])
    f_2_int = interpolator(coords_2)

    assert torch.min(torch.abs(f_2 - f_2_int)) < 1e-4, "Too large error"
