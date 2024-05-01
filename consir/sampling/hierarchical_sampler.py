import numpy as np
from consir.sampling import hierarchical_poisson_disc  

def sample_points_hierarchically(radius, domain=[(0, 1), (0, 1)], factor=np.sqrt(2.0)):
    """
    Perform hierarchical Poisson Disk sampling in a given domain, starting with a specified initial radius.

    This function generates points using Poisson Disk sampling starting from a specified radius, which 
    increases by a factor, by default sqrt(2), with each new level of sampling. Points are sampled hierarchically, 
    meaning that each new set of points is dependent on the position of the points from the previous set. 
    The function continues sampling until no new points are generated or only one new point is generated, 
    indicating that the domain space has been sufficiently filled at the current level of granularity.

    Parameters:
    radius (float): The initial radius for the Poisson Disk sampler.
    domain (list of tuple of float): The bounds of the sampling domain in each dimension, 
                                    where each tuple represents the minimum and maximum bounds along a dimension.
                                    Defaults to [(0, 1), (0, 1)] for a 2-dimensional unit square.
    factor (float): the growth radius of the disc size.

    Returns:
    tuple: A tuple containing two numpy arrays:
        - all_points_array (numpy.ndarray): A 2D array where each row represents the coordinates of a sampled point.
        - levels_array (numpy.ndarray): A 1D array where each element represents the hierarchical level of the corresponding point in `all_points_array`.
        
    Notes:
    - The function prints the current level and the number of points sampled at each iteration.
    """
    initial_sampler = hierarchical_poisson_disc.PoissonDiskSampler(r=radius, k=30, domain=domain)
    initial_set = initial_sampler.sample()

    # List to hold all points and their levels
    all_points = []
    levels = []

    # Append initial points and level
    all_points.extend(initial_set)
    levels.extend([0] * len(initial_set))  # Level 0 for all initial points

    current_radius = radius
    last_set = initial_set
    count = 0
    empty = False

    while not empty:
        count += 1
        current_radius *= factor  # Increase the radius
        sub_sampler = hierarchical_poisson_disc.PoissonDiskSamplingPrecomputed(
            last_set, r=current_radius, k=120, dimensions=len(domain), domain=domain
        )
        new_set, idx = sub_sampler.sample()
        print(count, len(new_set))

        # Append new points and their levels
        all_points.extend(new_set)
        levels.extend([count] * len(new_set))  # Append level count for each new point

        if len(new_set) <= 1:
            empty = True
        last_set = new_set  # Update the last set to the new set

    # Convert lists to NumPy arrays
    all_points_array = np.array(all_points)
    levels_array = np.array(levels)

    return all_points_array, levels_array