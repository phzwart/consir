import torch
import torch.nn.functional as F


class GridInterpolator:
    def __init__(self, grid, domain):
        """
        Initialize the GridInterpolator with a given grid and domain.

        Args:
            grid (torch.Tensor): A 2D tensor of shape (N, N) representing the grid.
            domain (list of tuples): The domain specified as [(a, b), (c, d)],
                                     where 'a' and 'b' are the min and max for the x-axis,
                                     and 'c' and 'd' are the min and max for the y-axis.
        """
        assert grid.ndim == 2, "Grid must be a 2D tensor."
        self.grid = grid.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, N, N)
        self.x_min, self.x_max = domain[0]
        self.y_min, self.y_max = domain[1]
        self.N = grid.shape[0]

    def __call__(self, points):
        """
        Interpolate the grid at specific points.

        Args:
            points (torch.Tensor): A tensor of shape (K, 2) where each row represents
                                   the (x, y) coordinates within the specified domain.

        Returns:
            torch.Tensor: Interpolated values at the given points, shaped (K, 1).
        """
        # Normalize points from the given domain to [0, 1]
        points[:, 0] = (points[:, 0] - self.x_min) / (self.x_max - self.x_min)
        points[:, 1] = (points[:, 1] - self.y_min) / (self.y_max - self.y_min)

        # Scale points from [0, 1] to [-1, 1] (required by grid_sample)
        points = 2 * points - 1

        points = points[:, [1, 0]]

        # Reshape points for grid_sample: (batch size, height, width, 2)
        points = points.unsqueeze(0).unsqueeze(1)  # Shape: (1, 1, K, 2)

        # Interpolate using grid_sample
        interpolated_values = F.grid_sample(
            self.grid, points, mode="bicubic", padding_mode="zeros", align_corners=True
        )

        # Remove unnecessary dimensions and return the result shaped (K, 1)
        return interpolated_values.squeeze().view(-1, 1)  # Shape: (K, 1)
