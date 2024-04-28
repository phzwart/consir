import numpy as np
import pytest
from consir.sampling.hierarchical_poisson_disc import PoissonDiskSampler


def test_initialization():
    sampler = PoissonDiskSampler(domain=[(0, 1), (0, 0.5)], r=0.1, k=30)
    assert sampler.r == 0.1
    assert sampler.k == 30
    assert np.array_equal(sampler.domain, np.array([(0, 1), (0, 0.5)]))


def test_sample_generation():
    sampler = PoissonDiskSampler(domain=[(0, 1), (0, 0.5)], r=0.05, k=10)
    samples = sampler.sample()
    assert len(samples) > 0
    for point in samples:
        assert 0 <= point[0] <= 1
        assert 0 <= point[1] <= 0.5


def test_minimum_distance():
    min_dist = 0.1
    safety = 0.70
    sampler = PoissonDiskSampler(domain=[(0, 1), (0, 0.5)], r=min_dist, k=30)
    samples = sampler.sample()
    distances = np.sqrt(
        ((samples[:, np.newaxis] - samples[np.newaxis, :]) ** 2).sum(axis=2)
    )
    np.fill_diagonal(distances, np.inf)
    assert np.all(distances >= min_dist * safety)


def test_edge_case_empty_domain():
    """
    Tests that the PoissonDiskSampler raises an AssertionError when initialized with a zero-volume domain.
    """
    with pytest.raises(AssertionError) as excinfo:
        sampler = PoissonDiskSampler(domain=[(0, 0), (0, 0)], r=0.1, k=30)
    assert "We need to have domain with non-zero volume / area" in str(
        excinfo.value
    ), "Error message for zero volume domain did not match."
