import numpy as np
import pytest
from consir.sampling.hierarchical_poisson_disc import PoissonDiskSamplingPrecomputed


def test_precomputed_initialization():
    points = np.random.uniform(0, 1, (100, 2))
    sampler = PoissonDiskSamplingPrecomputed(
        precomputed_points=points, r=0.1, k=30, dimensions=2, domain=[(0, 1), (0, 1)]
    )
    assert len(sampler.precomputed_points) == 100
    assert sampler.r == 0.1
    assert sampler.k == 30


def test_precomputed_sampling():
    points = np.random.uniform(0, 1, (100, 2))
    sampler = PoissonDiskSamplingPrecomputed(
        precomputed_points=points, r=0.05, k=10, dimensions=2, domain=[(0, 1), (0, 1)]
    )
    samples = sampler.sample()
    assert len(samples) > 0


def test_precomputed_minimum_distance():
    min_dist = 0.1
    safety = 0.70
    points = np.random.uniform(0, 1, (100, 2))
    sampler = PoissonDiskSamplingPrecomputed(
        precomputed_points=points,
        r=min_dist,
        k=120,
        dimensions=2,
        domain=[(0, 1), (0, 1)],
    )
    samples = sampler.sample()
    distances = np.sqrt(
        ((samples[:, np.newaxis] - samples[np.newaxis, :]) ** 2).sum(axis=2)
    )
    np.fill_diagonal(distances, np.inf)
    print(distances)
    print(np.min(distances))
    acceptable_min_dist = min_dist * safety
    assert np.all(distances >= acceptable_min_dist)


def test_precomputed_no_samples_possible():
    points = np.array([[0.5, 0.5]])
    sampler = PoissonDiskSamplingPrecomputed(
        precomputed_points=points, r=1, k=30, dimensions=2, domain=[(0, 1), (0, 1)]
    )
    samples = sampler.sample()
    assert len(samples) == 1  # Only the initial point, no room for more due to large r


def test_precomputed_out_of_bounds():
    min_dist = 0.1
    points = np.random.uniform(0, 1, (100, 2))
    sampler = PoissonDiskSamplingPrecomputed(
        precomputed_points=points,
        r=min_dist,
        k=120,
        dimensions=2,
        domain=[(0, 0.1), (0.1, 0.5)],
    )
    assert not sampler.is_valid_point(np.array((3.0, 3.0)))
