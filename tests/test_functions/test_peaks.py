import pytest
import torch
import numpy as np
import scipy.special as sp
from consir.functions.peaks import random_peaks, compute_mu, heteroscedastic_error

def test_random_peaks_basic():
    """Test the basic functionality of random_peaks."""
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), indexing='ij')
    x = torch.stack( [X.flatten(), Y.flatten()], dim=1 )
    result = random_peaks(x, num_peaks=5)
    print(result.shape)
    assert result.shape[0] == x.shape[0], "Output shape mismatch"
    assert torch.all(result >= 0), "All values in the result should be non-negative"

def test_random_peaks_max_value():
    """Test if the output of random_peaks is capped at the specified maximum value."""
    X, Y = torch.meshgrid(torch.linspace(-1, 1, 10), torch.linspace(-1, 1, 10), indexing='ij')
    x = torch.stack( [X.flatten(), Y.flatten()], dim=1 )
    maxi = 0.5
    result = random_peaks(x, num_peaks=5, maxi=maxi)
    assert torch.all(result <= maxi), "Values in the result should be capped at maxi"

def test_compute_mu():
    """Test the compute_mu function for known values."""
    k = 2
    expected_value = (2 ** 0.5) * (sp.gamma((k + 1) / 2) / sp.gamma(k / 2))
    computed_value = compute_mu(k)
    assert np.isclose(computed_value, expected_value), "compute_mu function returned incorrect value"

def test_heteroscedastic_error_student_t():
    """Test the heteroscedastic_error function with student_t distribution."""
    shape = [10000]
    noise = heteroscedastic_error(shape, error_type='student_t', df=100, variance=1.0)
    m = torch.mean(noise)
    s = torch.std(noise)
    assert abs(m) < 1e-2, "Mean is off"
    assert abs(s-1) < 1e-2, "Sigma is off"
    assert noise.shape[0] == shape[0], "Noise output shape should match mean shape"

def test_heteroscedastic_error_asymmetric():
    """Test the heteroscedastic_error function with chi_squared distribution."""
    shape = [1000]
    noise = heteroscedastic_error(shape, error_type='asymmetric', df=2, variance=1.0)
    assert noise.shape[-1] == shape[-1], "Noise output shape should match mean shape"
    m = torch.mean(noise)
    s = torch.std(noise)
    assert abs(m) < 1e-1, "Mean is off %s"%m
    assert abs(s-0.6551) < 1e-1, "Sigma is off %s"%s

def test_invalid_error_type():
    """Test the heteroscedastic_error function with an invalid error type."""
    mean = torch.zeros(10)
    with pytest.raises(ValueError, match="Error type not supported: Choose 'student_t' or 'asymmetric'"):
        heteroscedastic_error(mean.shape, error_type='invalid_type')
