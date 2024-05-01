import numpy as np


def partition_data(
    coords, values, levels, test_fraction, cal_fraction, N_test, level_of_selected_set
):
    """
    Partition the input data into a calibration set and multiple train/test sets based on the specified criteria.

    Parameters:
    - coords (np.ndarray): An array of coordinates, shape (N, ...), where N is the number of samples.
    - values (np.ndarray): An array of values corresponding to each coordinate, shape (N, ...).
    - levels (np.ndarray): An array of integer levels, shape (N,).
    - test_fraction (float): The fraction of data to be used as test set in each of the N_test partitions.
    - cal_fraction (float): The fraction of the total data to be used as the calibration set.
    - N_test (int): The number of test/train set pairs to generate.
    - level_of_selected_set (int or None): The specific level required for the calibration set. If None, levels are ignored.

    Returns:
    - cal_set_coords (np.ndarray): Coordinates of the calibration set.
    - cal_set_values (np.ndarray): Values of the calibration set.
    - train_test_sets (list): A list of tuples, each containing train and test data pairs. Each pair is a tuple
      (train_set, test_set), where each set is a tuple (coords, values).
    """
    assert (
        len(coords) == len(values) == len(levels)
    ), "All input arrays must have the same length."
    n_total = len(coords)
    assert (
        test_fraction + cal_fraction < 1
    ), "summed calibration and test fraction need to be less than 1"

    # Select indices for the calibration set
    if level_of_selected_set is None:
        cal_indices = np.random.choice(
            n_total, size=int(n_total * cal_fraction), replace=False
        )
    else:
        eligible_indices = np.where(levels == level_of_selected_set)[0]
        cal_indices = np.random.choice(
            eligible_indices,
            size=int(len(eligible_indices) * cal_fraction),
            replace=False,
        )

    # Extract calibration set
    cal_set_coords = coords[cal_indices]
    cal_set_values = values[cal_indices]

    # Indices remaining for train/test sets
    remaining_indices = np.array([i for i in range(n_total) if i not in cal_indices])

    # Generate N_test train/test partitions
    train_test_sets = []
    for _ in range(N_test):
        if level_of_selected_set is None:
            test_indices = np.random.choice(
                remaining_indices,
                size=int(len(remaining_indices) * test_fraction),
                replace=False,
            )
        else:
            eligible_test_indices = remaining_indices[
                np.in1d(levels[remaining_indices], level_of_selected_set)
            ]
            test_indices = np.random.choice(
                eligible_test_indices,
                size=int(len(eligible_test_indices) * test_fraction),
                replace=False,
            )

        train_indices = np.array(
            [i for i in remaining_indices if i not in test_indices]
        )

        train_set = (coords[train_indices], values[train_indices])
        test_set = (coords[test_indices], values[test_indices])
        train_test_sets.append((train_set, test_set))

    return cal_set_coords, cal_set_values, train_test_sets
