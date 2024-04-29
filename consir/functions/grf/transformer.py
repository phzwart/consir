import torch


def transform_grf(grf, kappa, alpha_quantile):

    grf = (grf - grf.mean()) / grf.std()  # normalize

    alpha = grf.quantile(
        alpha_quantile
    )  # determine the alpha level based on user input

    new_rho = 1 / (
        1 + torch.exp(-kappa * (grf - alpha))
    )  # apply sigmoid transform and calculate the correlation

    return new_rho
