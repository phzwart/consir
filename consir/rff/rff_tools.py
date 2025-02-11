import torch
import torch.nn as nn
import numpy as np

class RandomFourierFeatures(nn.Module):
    """
    Module to compute random Fourier features for kernel approximation.

    Args:
        input_dim (int): Dimension of the input data.
        output_dim (int): Number of random Fourier features to generate.
        gamma (float, optional): Parameter for the kernel. Default is 1.0.

    Attributes:
        W (torch.Tensor): Random weight matrix for feature projection.
        b (torch.Tensor): Random bias vector.

    """
    def __init__(self, input_dim, output_dim, gamma, df=3):
        super(RandomFourierFeatures, self).__init__()
        self.gamma = gamma
        #self.W = torch.randn(output_dim, input_dim) * np.sqrt(2 * self.gamma)
        Z = torch.randn(output_dim, input_dim)
        # Chi-squared samples using Gamma distribution
        k = df / 2.0
        theta = 2.0
        gamma_dist = torch.distributions.Gamma(k, theta)
        V = gamma_dist.sample((output_dim, input_dim))
        # Compute t-distributed samples
        self.W = np.sqrt(2 * self.gamma) * Z / torch.sqrt(V / df)

        self.b = torch.rand(output_dim) * 2 * np.pi

    def forward(self, x):
        """
        Forward pass to compute random Fourier features.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, input_dim).

        Returns:
            torch.Tensor: Transformed features of shape (batch_size, output_dim).

        """
        # Apply the Random Fourier Feature transformation
        projection = torch.matmul(x, self.W.T) + self.b
        return np.sqrt(2.0 / self.W.shape[0]) * torch.cos(projection)

class RFFCerberus(nn.Module):
    """
    Neural network module using random Fourier features for quantile regression.

    Args:
        input_dim (int): Dimension of the input data.
        latent_dim (int): Dimensionality of the latent space after random Fourier features.

    Attributes:
        rff (RandomFourierFeatures): Module to compute random Fourier features.
        median_head (nn.Linear): Linear layer to predict the median.
        lower_quantile_head (nn.Linear): Linear layer to predict the lower quantile.
        upper_quantile_head (nn.Linear): Linear layer to predict the upper quantile.

    """
    def __init__(self, input_dim, latent_dim, gamma, df=5, clip_low=None, clip_high=None):
        super(RFFCerberus, self).__init__()
        # Random Fourier Features module
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.clip_low = clip_low
        self.clip_high = clip_high

        self.rff = RandomFourierFeatures(self.input_dim, self.latent_dim, self.gamma, df)
        # Linear layers to map latent vector to quantile predictions
        self.median_head = nn.Linear(latent_dim, 1)
        self.lower_quantile_head = nn.Linear(latent_dim, 1)
        self.upper_quantile_head = nn.Linear(latent_dim, 1)
        self.sp = nn.Softplus()

    def forward(self, x):
        """
        Forward pass to compute quantile predictions.

        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, input_dim).

        Returns:
            tuple: A tuple containing:
                - torch.Tensor: Predicted lower quantile.
                - torch.Tensor: Predicted median.
                - torch.Tensor: Predicted upper quantile.

        """
        # Obtain latent representation
        latent = self.rff(x)

        # Pass through each head
        median = torch.clip(self.median_head(latent), self.clip_low, self.clip_high)
        lower = torch.clip( median - self.sp(self.lower_quantile_head(latent)), self.clip_low, self.clip_high)
        upper = torch.clip( median + self.sp(self.upper_quantile_head(latent)),self.clip_low, self.clip_high)
        return lower, median, upper

def pinball_loss(predictions, targets, quantile):
    """
    Computes the pinball loss for quantile regression.

    Args:
        predictions (torch.Tensor): Predicted quantiles.
        targets (torch.Tensor): Ground truth values.
        quantile (float): Quantile to compute the loss for (between 0 and 1).

    Returns:
        torch.Tensor: Scalar tensor representing the pinball loss.

    """
    errors = targets - predictions
    loss = torch.where(errors > 0, quantile * errors, (quantile - 1) * errors)
    return torch.mean(loss)
