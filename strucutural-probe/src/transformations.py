import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod


class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass


class RBFKernel(nn.Module):
    def __init__(self, in_features, out_features, sigma=None):
        super(). __init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centers = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigma = sigma if sigma is not None else 1.0
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.uniform_(self.centers, -1, 1)
    
    def forward(self, x):
        size = (x.size(0), self.out_features, self.in_features)
        x = x.unsqueeze(1).expand(size)
        centers = self.centers.unsqueeze(0).expand(size)
        distances = torch.sum((x - centers) ** 2, -1)
        return torch.exp(-distances / (2 * self.sigma ** 2))


class RBFKernelLayer(BaseModel):
    def __init__(self, input_dim: int, rbf_features: int, output_dim: int, rbf_sigma: float=1.0) -> None:
        super().__init__()
        self.rbf_kernel = RBFKernel(input_dim, rbf_features, sigma=rbf_sigma)
        self.linear = nn.Linear(rbf_features, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.rbf_kernel(x)
        x = self.linear(x)
        return x


class LinearTransformation(BaseModel):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class OrthogonalLayer(BaseModel):
    def __init__(self, input_dim):
        super(OrthogonalLayer, self).__init__()
        self.weights = nn.Parameter(torch.Tensor(input_dim, input_dim))
        nn.init.orthogonal_(self.weights)

    def forward(self, x) -> torch.Tensor:
        return F.linear(x, self.weights)


class MultilayerPerceptron(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        return x
