import numpy as np
import pytest

# Assuming `generator` and `properties` are parts of modules you've written or imported
from consir.functions.grf import generator, properties

# Constants
K = 128
COEFS = [0.68934415, 1.79430679, -1.25884068, -5.43033948]
POLYFIT = np.poly1d(COEFS)
THRESHOLD = 0.15
ALPHAS = np.linspace(-2, -1, 10)

@pytest.mark.parametrize("alpha", ALPHAS)
def test_gaussian_random_field(alpha):
    results = []
    for _ in range(50):
        tmp = generator.generate_grf(alpha=10**alpha, grid_size=(K)).numpy()
        cor_length = properties.compute_autocorrelation(tmp)
        results.append(cor_length)

    m = np.mean(np.array(results))
    s = np.std(np.array(results))
    assert abs(np.log(m) - POLYFIT(alpha)) < THRESHOLD, f"Failed at alpha={alpha}: {abs(np.log(m) - POLYFIT(alpha))}"

# Tests for find_baseline
def test_find_baseline_simple():
    curve = np.array([1, 2, 3, 4, 5, 3, 1])
    assert properties.find_baseline(curve) == (1 + 1) / 2.0, "Should calculate correct baseline for a simple peak"

def test_find_baseline_flat_curve():
    curve = np.ones(10)
    assert properties.find_baseline(curve) == 1.0, "Should return the flat value as baseline"

def test_find_baseline_no_transition():
    curve = np.array([1, 2, 3, 4, 5])
    with pytest.raises(AssertionError):
        properties.find_baseline(curve)

def test_peak_near_center():
    # Creating a curve with the peak near the center
    curve = np.array([1, 2, 3, 5, 7, 5, 3, 2, 1])
    expected_baseline = (1 + 1) / 2.0  # Calculating expected baseline
    assert properties.find_baseline(curve) == expected_baseline, "Baseline calculation is incorrect"

def test_peak_not_near_center():
    # Creating a curve with the peak not near the center
    curve = np.array([7, 5, 3, 2, 1, 2, 3, 5, 7])
    with pytest.raises(AssertionError):
        properties.find_baseline(curve)

def test_correct_baseline_calculation():
    # A curve with a clear baseline difference calculation
    curve = np.array([0, 0, 1, 3, 6, 8, 6, 3, 1, 0, 0])
    expected_baseline = (0 + 0) / 2.0  # Start and end of the curve are baseline
    assert properties.find_baseline(curve) == expected_baseline, "Baseline calculation failed to identify correct values"

# Tests for compute_autocorrelation
def test_compute_autocorrelation_standard():
    # Creating a 2D array with random values
    x = np.linspace(-1,1,64)
    X,Y = np.meshgrid(x,x)
    image = np.exp(-(X*X+Y*Y)/(2.0*0.01))
    assert properties.compute_autocorrelation(image) is not None, "Should compute a valid correlation length"

def test_compute_autocorrelation_non_square():
    image = np.random.rand(10, 9)
    with pytest.raises(AssertionError):
        properties.compute_autocorrelation(image)

def test_compute_autocorrelation_uniform():
    # Uniform value array should have autocorrelation at maximum initially
    image = np.ones((10, 10))
    assert properties.compute_autocorrelation(image) is None, "Should return None for uniform image as baseline exceeds peak"