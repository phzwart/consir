import torch


def rastrigin_function(coords, a=10):
    """
    Compute the Rastrigin function for a set of points in N-dimensional space.

    The Rastrigin function is a non-convex function used as a performance test problem for optimization algorithms.
    It is defined as:

    f(x) = A*n + sum(x_i^2 - A * cos(2 * pi * x_i))

    where:
    - A is a constant (usually 10)
    - x_i are the components of the input vector x
    - n is the dimensionality of the vector x

    Parameters:
        coords (torch.Tensor): A tensor of shape (M, N) where M is the number of points and
                               N is the dimensionality of each point.
        a (float, optional): Constant A in the Rastrigin function, defaults to 10.

    Returns:
        torch.Tensor: A tensor of Rastrigin function values for each point, shape (M,).
    """
    n = coords.shape[1]
    return a * n + torch.sum(coords**2 - a * torch.cos(2 * torch.pi * coords), dim=1)
