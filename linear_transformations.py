import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearTransformation(nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class OrthogonalLayer(nn.Module):
    def __init__(self, input_dim):
        super(OrthogonalLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.orthogonal_(self.weights)

    def forward(self, x) -> torch.Tensor:
        return F.linear(x, self.weights)
