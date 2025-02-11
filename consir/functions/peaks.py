import numpy as np
import torch
import scipy.special as sp


def random_peaks(x,
                 num_peaks=10,
                 sigma_range=[0.05, 0.1], factor=3,
                 maxi=0.95, power=1.5):
    """
    Generate a 2D field with randomly placed Gaussian peaks.

    Parameters:
    X (torch.Tensor): A tensor representing the x-coordinates.
    Y (torch.Tensor): A tensor representing the y-coordinates.
    num_peaks (int): The number of random peaks to generate.
    sigma_range (list): A list containing the minimum and maximum range for the Gaussian sigma values.
    factor (float): A factor to ensure the peaks are within the bounds of the coordinate space.
    maxi (float): Maximum value allowed for any peak.
    power (float): The exponent applied to each Gaussian for shaping the peaks.

    Returns:
    torch.Tensor: A tensor containing the generated 2D field with peaks.
    """

    X = x[:,0]
    Y = x[:,1]

    locs = np.random.uniform(-1 + sigma_range[-1] * factor,
                             1 - sigma_range[-1] * factor, (num_peaks, 2))
    heights = np.random.uniform(-1, 1, (num_peaks, 2))
    heights = np.sqrt(np.sum(heights ** 2, axis=-1))
    sigmas = np.random.uniform(sigma_range[0], sigma_range[1], num_peaks)

    params = {}
    params['locs'] = locs
    params['heights'] = heights
    params['sigmas'] = sigmas

    result = torch.zeros_like(X)

    for l, h, s in zip(locs, heights, sigmas):
        tmp = ((X - l[0]) ** 2 / (2.0 * s * s)) + (
                    (Y - l[1]) ** 2 / (2.0 * s * s))
        tmp = torch.exp(-tmp)
        result += tmp ** power

    # Clip values to a maximum threshold
    sel = result > maxi
    result[sel] = maxi

    return result, params


def compute_mu(k):
    """
    Compute the mean for the square root of a chi-squared distribution.

    Parameters:
    k (int): Degrees of freedom for the chi-squared distribution.

    Returns:
    float: The mean value of the distribution.
    """
    return (2 ** 0.5) * (sp.gamma((k + 1) / 2) / sp.gamma(k / 2))


def heteroscedastic_error(shape, error_type='student_t', df=5, variance=0.1):
    """
    Generate heteroscedastic noise based on the given mean.

    Parameters:
    shape (tuple): The shape of the function to which noise is to be added.
    error_type (str): The type of noise distribution to use ('student_t' or 'chi_squared').
    df (int): Degrees of freedom for the error distribution.
    variance (float): The variance scaling factor for the noise.

    Returns:
    torch.Tensor: A tensor containing the generated noise.
    """
    if error_type == 'student_t':
        # Student-t distributed errors
        noise = variance * torch.distributions.StudentT(df).sample(shape)
    elif error_type == 'asymmetric':
        # sqrt(chi-square) distributed errors
        mu = compute_mu(df)
        noise = variance * torch.sqrt(
            torch.distributions.Chi2(df).sample(shape))
        noise = noise - variance * mu
    else:
        raise ValueError(
            "Error type not supported: Choose 'student_t' or 'asymmetric'")

    return noise
