import numpy as np
from consir.training.data_splitter import *
import pytest


def test_basic_functionality():
    """Test basic partitioning with typical inputs."""
    coords = np.random.rand(100, 2)
    values = np.random.rand(100)
    levels = np.ones(100)
    levels[50:] = 0
    test_fraction = 0.2
    cal_fraction = 0.1
    N_test = 5
    level_of_selected_set = 1

    cal_coords, cal_values, train_test_sets = partition_data(
        coords,
        values,
        levels,
        test_fraction,
        cal_fraction,
        N_test,
        level_of_selected_set,
    )

    # Test sizes of sets
    assert len(cal_coords) == int(0.1 * 50), "Calibration set size is incorrect"
    assert len(cal_values) == int(0.1 * 50), "Calibration set size is incorrect"

    for train_set, test_set in train_test_sets:
        assert len(test_set[0]) == int(
            0.2 * 45
        ), "Test set size is incorrect"  # 90 is 100 - 10% calibration


def test_with_no_level_specification():
    """Test the function without specifying a level for calibration set."""
    coords = np.random.rand(50, 2)
    values = np.random.rand(50)
    levels = np.ones(50)
    levels[25:] = 0

    test_fraction = 0.1
    cal_fraction = 0.2
    N_test = 3
    level_of_selected_set = None

    cal_coords, cal_values, train_test_sets = partition_data(
        coords,
        values,
        levels,
        test_fraction,
        cal_fraction,
        N_test,
        level_of_selected_set,
    )

    # Calibration set should still be 20% of the total data
    assert len(cal_coords) == int(0.2 * 50), "Calibration set size is incorrect"
    assert len(cal_values) == int(0.2 * 50), "Calibration set size is incorrect"


def test_input_validation():
    """Test the function's ability to handle incorrect inputs."""
    coords = np.random.rand(100, 2)
    values = np.random.rand(100)
    levels = np.random.randint(0, 3, size=99)  # Incorrect size

    with pytest.raises(AssertionError):
        partition_data(coords, values, levels, 0.1, 0.2, 5, 1)


def test_fraction_sum_validation():
    """Test that an assertion error is raised when test_fraction and cal_fraction sum to 1 or more."""
    coords = np.random.rand(30, 2)
    values = np.random.rand(30)
    levels = np.random.randint(0, 3, size=30)
    test_fraction = 0.5
    cal_fraction = 0.5  # This sums to 1.0 with test_fraction
    N_test = 1
    level_of_selected_set = None

    with pytest.raises(
        AssertionError,
        match="summed calibration and test fraction need to be less than 1",
    ):
        partition_data(
            coords,
            values,
            levels,
            test_fraction,
            cal_fraction,
            N_test,
            level_of_selected_set,
        )
