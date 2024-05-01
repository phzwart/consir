import numpy as np
import pytest
from consir.sampling import hierarchical_sampler


def test_shift_labels_basic():
    """Test that shift_labels correctly shifts labels in a simple array."""
    input_array = np.array([0, 0, 0, 2, 0, 3, 0, 4, 4])
    expected_output = np.array([0, 0, 0, 1, 0, 2, 0, 3, 3])
    assert np.array_equal(hierarchical_sampler.shift_labels(input_array), expected_output)

def test_basic_functionality():
    radius = 0.01
    points, levels = hierarchical_sampler.sample_points_hierarchically(radius)
    assert isinstance(points, np.ndarray), "Points should be a NumPy array"
    assert isinstance(levels, np.ndarray), "Levels should be a NumPy array"
    assert points.ndim == 2, "Points array should be 2D (points x dimensions)"
    assert points.shape[0] == levels.shape[0], "Each point should have a corresponding level"
    assert points.shape[0] > 0, "Should generate at least one point"
    assert levels.min() == 0, "Initial points should be at level 0"

def test_correctness_of_levels():
    radius = 0.1
    points, levels = hierarchical_sampler.sample_points_hierarchically(radius)
    # Check if levels increment correctly
    unique_levels = np.unique(levels)
    assert np.all(np.diff(unique_levels) == 1), "Levels should increment by 1"

def test_different_domains():
    radius = 0.1
    domains = [
        [(0, 1), (0, 1)],  # 2D
        [(0, 1), (0, 1), (0, 1)],  # 3D
        [(0, 0.5), (0, 0.5)]  # Smaller domain in 2D
    ]
    for domain in domains:
        points, levels = hierarchical_sampler.sample_points_hierarchically(radius, domain=domain)
        assert points.shape[1] == len(domain), f"Dimension mismatch for domain {domain}"
