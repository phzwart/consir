import torch


def generate_grf(grid_size, alpha, device="cpu", eps=1e-3):
    linspaced = torch.linspace(
        -1, 1, steps=grid_size, device=device
    )  # create a linear spaced grid of even spaced points between -1 and 1 with 'grid_size' number of points

    x, y = torch.meshgrid(linspaced, linspaced, indexing="ij")  # convert into 2D grid

    distanced_sqrd = (
        x**2 + y**2
    )  # calculate the squared distance of each point in the grid from the origin

    covariance_matrix = torch.exp(
        -distanced_sqrd / (2 * alpha**2)
    )  # create a covariance matrix based on the squared distances. Alpha will determine the decay rate
    covariance_matrix = torch.fft.fftshift(covariance_matrix)

    noise = torch.randn(
        grid_size, grid_size, device=device
    )  # randomly generate a grid of standard normal noise

    fft_noise = torch.fft.fft2(
        noise
    )  # run a fourier transform on the noise to convert it into the frequency domain

    grf = torch.fft.ifft2(
        fft_noise * covariance_matrix
    ).real  # modify the amp. of the frequencies based on the covariance, then transform the modfied frequency domain back to spatial

    assert torch.std(grf) > eps, "A flat image was generated. Increase alpha"

    return grf
