import numpy as np
from ..sampling.topdown.poissontiler import PoissonTiler

def generate_hierarchical_points(area, 
                                 tile_size=20, 
                               sampling_distance=10,
                               final_distance=3,
                               scale_factor=np.sqrt(3),
                               n_processes=None):
    """
    Generate hierarchical Poisson disk samples across an area with geometrically decreasing distances.
    
    Args:
        area (tuple): ((min_x, max_x), (min_y, max_y)) defining the area to cover
        tile_size (float): Size of the square tile
        sampling_distance (float, optional): Initial/largest sampling distance. Defaults to 3.0.
        final_distance (float, optional): Smallest sampling distance. Defaults to sampling_distance/3.
        scale_factor (float, optional): Factor by which sampling distance decreases. Defaults to sqrt(3).
        n_processes (int, optional): Number of processes for parallel computation. Defaults to None.
    
    Returns:
        tuple: (points, levels) - Arrays of points and their corresponding sampling levels
    """
    if final_distance is None:
        final_distance = sampling_distance / 3.0
    
    # Generate sequence of sampling distances
    distances = []
    current_distance = sampling_distance
    while current_distance >= final_distance:
        distances.append(current_distance)
        current_distance /= scale_factor
    
    # Create tiler with the sequence of distances
    tiler = PoissonTiler(tile_size=tile_size, spacings=distances)
    
    # Generate points across the specified area
    points, levels = tiler.get_points_in_region(area, n_processes=n_processes)
    
    return points, levels
