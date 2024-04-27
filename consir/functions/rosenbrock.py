import torch


def rosenbrock_function(coords, a=1, b=100):
    """
    Compute the Rosenbrock function for a set of points in N-dimensional space.

    The Rosenbrock function is a complex mathematical function used for testing gradient-based optimization algorithms.
    It is characterized by a large, flat valley and a curved, narrow ridge; hence, it also refered to as Rosenbrock's valley or Rosenbrock's banana function.
    It is defined as:

    f(x) = sum(100 * (x_{i+1} - x_i^2)^2 + (a - x_i)^2) for i = 1 to n-1

    where:
    - x_i are the components of the input vector x
    - a and b are parameters of the function

    Parameters:
        coords (torch.Tensor): A tensor of shape (M, N) where M is the number of points and
                               N is the dimensionality of each point.
        a (float, optional): Parameter a of the Rosenbrock function, defaults to 1.
        b (float, optional): Parameter b of the Rosenbrock function, defaults to 100.

    Returns:
        torch.Tensor: A tensor of Rosenbrock function values for each point, shape (M,).
    """
    return torch.sum(
        b * (coords[:, 1:] - coords[:, :-1] ** 2) ** 2 + (a - coords[:, :-1]) ** 2,
        dim=1,
    )
