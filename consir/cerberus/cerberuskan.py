import torch
import torch.nn as nn
import random

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np

class Cerberus_KAN(nn.Module):
    def __init__(self,
                 n_input,
                 n_latent,
                 n_output,
                 base,
                 project,
                 knots_base = 5,
                 knots_head = 2
                 ):
        """
        Cerberus-KAN architecture with one base MLP-KAN producing a feature vector,
        and three heads (MLP-KANs) attached to the feature vector.

        Args:
        - n_input: Number of input features for the base MLP-KAN.
        - n_hidden: Number of hidden units in each MLP-KAN layer.
        - feature_size: The size of the feature vector output by the base MLP-KAN.
        - n_layers_phi: Number of hidden layers in the phi_j networks.
        - n_layers_g: Number of hidden layers in the g_i networks.
        """
        super(Cerberus_KAN, self).__init__()

        self.knots_base = knots_base
        self.knots_head = knots_head
        base.append(n_latent)
        project.append(n_output)
        self.base_kan = DeepKAN(n_input, base, num_knots=self.knots_base)
        self.head_1 = DeepKAN(n_latent, project, num_knots=self.knots_head)
        self.head_2 = DeepKAN(n_latent, project, num_knots=self.knots_head)
        self.head_3 = DeepKAN(n_latent, project, num_knots=self.knots_head)
        self.sp = nn.Softplus()
        self.tanh =  nn.Tanh()

    def forward(self, x):
        """
        Forward pass for Cerberus-KAN.

        Args:
        - x: Input tensor of shape (batch_size, n_input).

        Returns:
        - Three separate scalar outputs from the three heads.
        """
        feature_vector = self.tanh(self.base_kan(x))
        # Pass the feature vector through the three heads
        output_1 = self.sp(self.head_1(feature_vector))
        output_2 = self.head_2(feature_vector)
        output_3 = self.sp(self.head_3(feature_vector))
        return output_2 - output_1, output_2, output_2 + output_3