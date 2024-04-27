import torch


def ackley_function(coords, a=20, b=0.2, c=2 * torch.pi):
    """
    Compute the Ackley function for a set of points in N-dimensional space.

    The Ackley function is a complex mathematical function used for testing optimization algorithms.
    It is characterized by a nearly flat outer region and a large hole at the centre. The function
    is defined as:

    f(x) = -a * exp(-b * sqrt(1/n * sum(x_i^2))) - exp(1/n * sum(cos(c * x_i))) + a + exp(1)

    where:
    - x_i are the components of the input vector x
    - a, b, and c are parameters of the function
    - n is the dimensionality of the vector x

    Parameters:
        coords (torch.Tensor): A tensor of shape (M, N) where M is the number of points and
                               N is the dimensionality of each point.
        a (float, optional): Parameter of the Ackley function, defaults to 20.
        b (float, optional): Parameter of the Ackley function, influencing the exponential
                             decay rate, defaults to 0.2.
        c (float, optional): Parameter of the Ackley function, influencing the frequency
                             of the cosine term, defaults to 2*pi.

    Returns:
        torch.Tensor: A tensor of Ackley function values for each point, shape (M,).
    """
    # First term of the function
    first_term = -a * torch.exp(-b * torch.sqrt(torch.mean(coords**2, dim=1)))
    # Second term of the function
    second_term = -torch.exp(torch.mean(torch.cos(c * coords), dim=1))
    # The Ackley function
    result = first_term + second_term + a + torch.exp(torch.tensor(1.0))
    return result
