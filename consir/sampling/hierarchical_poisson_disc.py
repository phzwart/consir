"""
Here is code for creating hierarchical poisson-disc sampling point sets

https://www.cs.ubc.ca/~rbridson/docs/bridson-siggraph07-poissondisk.pdf

We use Bridson sampling to generate a point set in the PoissonDiskSampler class.
Subsequent use of that point set in the PoissonDiskSamplerPrecomputed class allows you to get a sample
that is also a Poisson Disk Sampled, but at a higher radius. This process can be iterated to build a set
of hierarchical points.

"""

import numpy as np
from scipy.spatial import KDTree


def compute_domain_volume(domain):
    """
    Computes the volume of a multidimensional domain specified as a list of tuples.
    Each tuple contains the minimum and maximum bounds for that dimension.

    Args:
        domain (list of tuples): Domain specification, where each tuple (min, max)
                                 represents the bounds along a dimension.

    Returns:
        float: The computed volume of the domain.
    """
    volume = 1
    for dim in domain:
        min_bound, max_bound = dim
        volume *= (
            max_bound - min_bound
        )  # Compute the length of each dimension and multiply

    return volume


class PoissonDiskSampler(object):
    """
    A class to generate samples using Poisson Disk Sampling within a specified domain.

    Attributes:
        domain (list of tuples): Boundaries for each dimension in the domain.
        r (float): Minimum distance between samples.
        k (int): Maximum number of attempts to generate a new sample around each existing sample.
        cell_size (float): Size of the cell in the grid, computed as r divided by the square root of the number of dimensions.
    """

    def __init__(self, domain, r, k=60):
        """
        Initializes the PoissonDiskSampler with the given domain, minimum distance, and optional parameters.

        Args:
            domain (list of tuples): Boundaries for each dimension in the domain, as (min, max) pairs.
            r (float): Minimum distance between samples.
            k (int, optional): Maximum number of attempts to generate a new sample. Defaults to 60.
        """
        self.domain = np.array(domain)
        assert (
            compute_domain_volume(self.domain) > 0
        ), "We need to have domain with non-zero volume / area"
        self.r = r
        self.k = k
        self.dimensions = len(domain)
        self.cell_size = r / np.sqrt(self.dimensions)

    def initialize_grid(self):
        """
        Initializes the grid where each cell can hold a sample point.

        Returns:
            ndarray: A grid initialized with -1, indicating empty cells.
        """
        domain_size = self.domain[:, 1] - self.domain[:, 0]
        grid_shape = np.ceil(domain_size / self.cell_size).astype(int)
        return np.full(grid_shape, -1, dtype=int)

    def to_grid_coords(self, point):
        """
        Converts a point from domain coordinates to grid coordinates.

        Args:
            point (array-like): The point in domain coordinates.

        Returns:
            tuple: The corresponding grid coordinates as a tuple.
        """
        return tuple(((point - self.domain[:, 0]) / self.cell_size).astype(int))

    def insert_sample(self, grid, point, index):
        """
        Inserts a sample point into the grid.

        Args:
            grid (ndarray): The grid to update.
            point (array-like): The sample point.
            index (int): The index of the sample in the samples list.
        """
        grid[self.to_grid_coords(point)] = index

    def generate_points_around(self, point):
        """
        Generates potential points around a given sample within the allowed radius.

        Args:
            point (array-like): The point around which to generate new points.

        Returns:
            ndarray: Array of new points around the given point.
        """
        radius = np.sqrt(np.random.uniform(self.r**2, (2 * self.r) ** 2, self.k))
        directions = np.random.normal(0, 1, (self.k, self.dimensions))
        unit_vectors = directions / np.linalg.norm(directions, axis=1)[:, None]
        return point + radius[:, None] * unit_vectors

    def is_valid_point(self, point, grid, idx_to_point):
        """
        Checks if a point is valid by ensuring it doesn't violate the minimum distance rule.

        Args:
            point (array-like): The point to check.
            grid (ndarray): The grid containing points.
            idx_to_point (dict): Mapping of grid indices to points.

        Returns:
            bool: True if the point is valid, False otherwise.
        """
        if np.any(point < self.domain[:, 0]) or np.any(point >= self.domain[:, 1]):
            return False
        grid_coords = self.to_grid_coords(point)
        slices = tuple(
            slice(max(0, gc - 1), min(d, gc + 2))
            for gc, d in zip(grid_coords, grid.shape)
        )
        for idx in np.nditer(grid[slices]):
            if idx >= 0:
                other_point = idx_to_point[int(idx)]
                if np.linalg.norm(point - other_point) < self.r:
                    return False
        return True

    def sample(self):
        """
        Generates a sample of points using the Poisson Disk Sampling method.

        Returns:
            ndarray: An array of sampled points.
        """
        grid = self.initialize_grid()
        initial_point = np.random.uniform(
            self.domain[:, 0], self.domain[:, 1], self.dimensions
        )
        samples = [initial_point]
        active_list = [0]
        idx_to_point = {0: initial_point}
        self.insert_sample(grid, initial_point, 0)

        while active_list:
            i = np.random.choice(active_list)
            current_point = samples[i]
            new_points = self.generate_points_around(current_point)

            valid_found = False
            for point in new_points:
                if self.is_valid_point(point, grid, idx_to_point):
                    samples.append(point)
                    new_index = len(samples) - 1
                    self.insert_sample(grid, point, new_index)
                    active_list.append(new_index)
                    idx_to_point[new_index] = point
                    valid_found = True
                    break

            if not valid_found:
                active_list.remove(i)

        return np.array(samples)


class PoissonDiskSamplingPrecomputed(object):
    """
    A class to perform Poisson Disk Sampling using a set of precomputed points within a specified non-uniform domain.

    Attributes:
        precomputed_points (ndarray): Precomputed points which are candidates for sampling.
        r (float): Minimum distance between samples.
        k (int): Maximum number of new points to consider around each sample point.
        dimensions (int): Number of dimensions in the domain.
        domain (array): Boundaries for each dimension in the domain, specified as (min, max) pairs.
        cell_size (float): Size of the cell in the grid, computed as r divided by the square root of the number of dimensions.
        kdtree (KDTree): KD-tree for efficient spatial queries.
        grid (ndarray): Spatial grid that helps in quick lookup of sample points.

        Due to inaccuracies - unclear where exactly - there is a possibility that the minimum distance is less than the
        the radius given. In tests, this is at about 80% of the specified given radius. At this point, this is not a
        concern but can be addressed at a later stage.
    """

    def __init__(self, precomputed_points, r, k, dimensions, domain):
        """
        Initializes the PoissonDiskSamplingPrecomputed with a set of precomputed points and other parameters.

        Args:
            precomputed_points (ndarray): Array of precomputed points within the domain.
            r (float): Minimum distance between samples.
            k (int): Maximum number of attempts to generate new points.
            dimensions (int): Number of dimensions in the domain.
            domain (list of tuples): Domain size for each dimension as (min, max) tuples.
        """
        self.precomputed_points = precomputed_points
        self.r = r
        self.k = k
        self.dimensions = dimensions
        self.domain = np.array(domain)
        assert (
            compute_domain_volume(self.domain) > 0
        ), "We need to have domain with non-zero volume / area"
        self.cell_size = r / np.sqrt(dimensions)
        self.kdtree = KDTree(precomputed_points)
        self.initialize_grid()

    def initialize_grid(self):
        """
        Initializes the spatial grid based on the domain size and cell size.
        This grid helps in quick lookup and spatial operations to ensure minimum distance constraints.
        """
        grid_shape = (
            np.ceil((self.domain[:, 1] - self.domain[:, 0]) / self.cell_size).astype(
                int
            )
            + 1
        )
        self.grid = np.full(grid_shape, -1, dtype=int)

    def to_grid_coords(self, point):
        """
        Converts a point from domain coordinates to grid coordinates.

        Args:
            point (array-like): The point in domain coordinates.

        Returns:
            tuple: The corresponding grid coordinates as a tuple, ensuring all values are within grid bounds.
        """
        coords = np.floor((point - self.domain[:, 0]) / self.cell_size).astype(int)
        max_bounds = np.array(self.grid.shape) - 1
        return tuple(coords.clip(0, max_bounds))

    def insert_sample(self, point, index):
        """
        Inserts a sample point into the grid.

        Args:
            point (array-like): The sample point.
            index (int): The index of the sample in the samples list.
        """
        self.grid[self.to_grid_coords(point)] = index

    def find_points_in_annulus(self, point, r, k):
        """
        Finds up to k points within the annulus of radii [r, 2r] around a given point.

        Args:
            point (array-like): The center point of the annulus.
            r (float): The inner radius of the annulus.
            k (int): Maximum number of points to return.

        Returns:
            ndarray: Points within the specified annulus.
        """
        indices = self.kdtree.query_ball_point(point, 2 * r)
        valid_indices = [
            idx
            for idx in indices
            if r <= np.linalg.norm(point - self.kdtree.data[idx]) <= 2 * r
        ]
        return self.kdtree.data[valid_indices][:k]

    def is_valid_point(self, point):
        """
        Checks if a point is valid by ensuring it does not violate the minimum distance constraint within its neighborhood.

        Args:
            point (array-like): The point to check.

        Returns:
            bool: True if the point is valid, False otherwise.
        """
        if any(point < self.domain[:, 0]) or any(point >= self.domain[:, 1]):
            return False
        grid_coords = self.to_grid_coords(point)
        slices = tuple(
            slice(max(0, gc - 1), min(gs, gc + 2))
            for gc, gs in zip(grid_coords, self.grid.shape)
        )
        for idx in np.nditer(self.grid[slices]):
            if idx >= 0:
                other_point = self.idx_to_point[int(idx)]
                if np.linalg.norm(point - other_point) < self.r:
                    return False
        return True
    
    def sample(self):
        """
        Generates a sample of points using the precomputed points and Poisson Disk Sampling method.
        Each call to this method starts with a fresh state, ensuring independent sampling runs.

        Returns:
            ndarray: An array of sampled points.
            list: List of indices corresponding to the original precomputed points.
        """
        self.initialize_grid()
        self.samples = []
        self.idx_to_point = {}
        original_indices = []  # List to store indices of the original precomputed points

        initial_index = np.random.randint(len(self.precomputed_points))
        initial_point = self.precomputed_points[initial_index]
        self.samples = [initial_point]
        self.idx_to_point = {0: initial_point}
        original_indices.append(initial_index)  # Append the index of the initial point
        self.insert_sample(initial_point, 0)

        active_list = [0]

        while active_list:
            i = np.random.choice(active_list)
            current_point = self.samples[i]
            new_points = self.find_points_in_annulus(current_point, self.r, self.k)

            valid_found = False
            for point in new_points:
                if (
                    all(point >= self.domain[:, 0]) and
                    all(point < self.domain[:, 1]) and
                    self.is_valid_point(point)
                ):
                    self.samples.append(point)
                    new_index = len(self.samples) - 1
                    self.insert_sample(point, new_index)
                    active_list.append(new_index)
                    self.idx_to_point[new_index] = point
                    # Find the index of this new point in the original precomputed list
                    original_idx = np.where((self.precomputed_points == point).all(axis=1))[0][0]
                    original_indices.append(original_idx)
                    valid_found = True
                    break

            if not valid_found:
                active_list.remove(i)

        return np.array(self.samples), original_indices
