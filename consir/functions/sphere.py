import torch


def sphere(coords):
    """
    Compute the Sphere function for a set of points in N-dimensional space.

    The Sphere function is a complex mathematical function used for testing optimization algorithms.
    This is as simple as it gets. The Sphere function is defined as:

    f(x) = sum(x_i^2)

    where x_i are the components of the input vector x.

    Parameters:
        coords (torch.Tensor): A tensor of shape (M, N) where M is the number of points and
                               N is the dimensionality of each point.

    Returns:
        torch.Tensor: A tensor of Sphere function values for each point, shape (M,).
    """
    return torch.sum(coords**2, dim=1)
