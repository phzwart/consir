import torch

class DataEngine:
    def __init__(self, data_tensor):
        """
        Initialize the DataEngine with a given tensor.

        Args:
            data_tensor (torch.Tensor): A tensor of shape (C, Y, X).
        """
        self.data_tensor = data_tensor

    def __call__(self, query_points):
        """
        Interrogate the data engine with query points to get nearest neighbor interpolated values.

        Args:
            query_points (torch.Tensor): A tensor of shape (K, 2) where each row contains (yi, xi).

        Returns:
            torch.Tensor: A tensor of shape (K, C) with nearest neighbor elements.
        """
        K = query_points.shape[0]
        C, Y, X = self.data_tensor.shape
        result = torch.empty((K, C))

        for i in range(K):
            yi, xi = query_points[i]
            # Nearest neighbor indices
            yi_nn = min(max(int(round(yi.item())), 0), Y - 1)
            xi_nn = min(max(int(round(xi.item())), 0), X - 1)

            # Get the nearest neighbor value
            result[i] = self.data_tensor[:, yi_nn, xi_nn]

        return result
