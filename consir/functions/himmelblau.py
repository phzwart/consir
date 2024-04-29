import torch


def himmelblau_function(x):
    """
    Calculate the Himmelblau function for a batch of points.

    Parameters:
    x (torch.Tensor): A 2D tensor of shape (n, 2) where each row represents a point (x, y).

    Returns:
    torch.Tensor: The function values of the Himmelblau function at each input point.
    """
    x1 = x[:, 0]
    y1 = x[:, 1]
    return (x1**2 + y1 - 11) ** 2 + (x1 + y1**2 - 7) ** 2
