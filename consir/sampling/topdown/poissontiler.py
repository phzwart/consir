import numpy as np
from .poissondiscsampling import PoissonDiskSamplerWithExisting
from multiprocessing import Pool
from functools import partial

class PoissonTiler:
    """
    Creates hierarchical Poisson disc sampling patterns that can be tiled across a large area.
    """
    def __init__(self, tile_size, spacings):
        """
        Initialize the tiler with tile size and spacing levels.
        
        Args:
            tile_size (float): Size of the square tile
            spacings (list): List of inter-point distances, from largest to smallest
        """
        self.tile_size = tile_size
        self.spacings = sorted(spacings, reverse=True)  # Ensure largest spacing first
        self.tile_domain = [(0, tile_size), (0, tile_size)]
        self.tile_points = None
        self.tile_labels = None
        
        # Generate the base tile
        self._generate_base_tile()

    def _generate_base_tile(self):
        """Generate hierarchical sampling within a single periodic tile."""
        points = None
        labels = None
        
        # Generate points for each spacing level
        for level, spacing in enumerate(self.spacings):
            sampler = PoissonDiskSamplerWithExisting(
                domain=self.tile_domain,
                r=spacing,
                existing_points=points,
                existing_labels=labels,
                wrap=True  # Enable periodic boundary conditions
            )
            
            points, labels = sampler.sample(new_label=level)
        
        self.tile_points = points
        self.tile_labels = labels

    def _process_tile(self, args):
        """Process a single tile - used for parallel processing."""
        ix, iy, min_x, min_y, max_x, max_y = args
        
        offset = np.array([ix * self.tile_size + min_x, 
                          iy * self.tile_size + min_y])
        
        tile_points = self.tile_points + offset
        
        mask = ((tile_points[:, 0] >= min_x) & 
                (tile_points[:, 0] < max_x) & 
                (tile_points[:, 1] >= min_y) & 
                (tile_points[:, 1] < max_y))
        
        return tile_points[mask], self.tile_labels[mask]

    def get_points_in_region(self, region, n_processes=None):
        """
        Get all points and their levels within a specified region using parallel processing.
        
        Args:
            region (tuple): ((min_x, max_x), (min_y, max_y)) defining the area to cover
            n_processes (int, optional): Number of processes to use. Defaults to None (CPU count)
        
        Returns:
            tuple: (points, labels) - Arrays of points and their corresponding sampling levels
        """
        (min_x, max_x), (min_y, max_y) = region
        
        nx = int(np.ceil((max_x - min_x) / self.tile_size))
        ny = int(np.ceil((max_y - min_y) / self.tile_size))
        
        # Prepare arguments for parallel processing
        tile_args = [
            (ix, iy, min_x, min_y, max_x, max_y)
            for ix in range(nx)
            for iy in range(ny)
        ]
        
        # Process tiles in parallel
        with Pool(processes=n_processes) as pool:
            results = pool.map(self._process_tile, tile_args)
        
        # Combine results
        all_points = []
        all_labels = []
        for points, labels in results:
            if len(points) > 0:
                all_points.extend(points)
                all_labels.extend(labels)
        
        return np.array(all_points), np.array(all_labels)
