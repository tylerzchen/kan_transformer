import torch
import torch.nn as nn


class RationalFunction(nn.Module):
    """rational activation function"""
    
    def __init__(self, degree_p=5, degree_q=4):
        """
        Args:
            degree_p (int): Degree of the numerator polynomial.
            degree_q (int): Degree of the denominator polynomial.
        """
        super().__init__()
        self.degree_p = degree_p
        self.degree_q = degree_q

        # Learnable coefficients for numerator (P) and denominator (Q)
        self.a = nn.Parameter(torch.randn(degree_p + 1))  # P(x) coefficients
        self.b = nn.Parameter(torch.randn(degree_q))      # Q(x) coefficients

    def forward(self, x):
        """Forward pass of the rational function."""
        # Compute P(x) = a_0 + a_1*x + a_2*x^2 + ...
        p = sum(self.a[i] * x**i for i in range(self.degree_p + 1))

        # Compute Q(x) = 1 + |b_0*x| + |b_1*x^2| + ...
        q = 1 + sum(torch.abs(self.b[i]) * torch.abs(x)**(i + 1) for i in range(self.degree_q))

        # Return R(x) = P(x) / Q(x)
        return p / q


class KAN(nn.Module):
    """Kolmogorov-Arnold Neural Network with Rational Base Functions."""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_cfg=dict(type="Rational", degree_p=5, degree_q=4),
        bias=True,
        drop=0.0,
    ):
        """
        Args:
            in_features (int): Number of input features.
            hidden_features (int, optional): Number of hidden features. Defaults to `in_features`.
            out_features (int, optional): Number of output features. Defaults to `in_features`.
            act_cfg (dict): Configuration for the rational activation. Includes degrees of P and Q.
            bias (bool): Whether to include bias in linear layers.
            drop (float): Dropout probability.
        """
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        # First fully connected layer
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)

        # First rational activation function
        self.act1 = RationalFunction(
            degree_p=act_cfg.get("degree_p", 5),
            degree_q=act_cfg.get("degree_q", 4)
        )

        # Dropout after first rational activation
        self.drop1 = nn.Dropout(drop)

        # Second fully connected layer
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        # Second rational activation function
        self.act2 = RationalFunction(
            degree_p=act_cfg.get("degree_p", 5),
            degree_q=act_cfg.get("degree_q", 4)
        )

        # Dropout after second rational activation
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        """Forward pass of the network."""
        x = self.fc1(x)
        x = self.act1(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.act2(x)
        x = self.drop2(x)
        return x