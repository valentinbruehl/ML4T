import torch
from torch import nn
from torch.nn import functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# let's use 3 hidden layers

class SimpleModel(nn.Module):
    def __init__(self, D_i, D_k, D_o) -> None:
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(D_i, D_k),
            nn.LeakyReLU(),
            nn.Linear(D_k, D_k),
            nn.LeakyReLU(),
            nn.Linear(D_k, D_k),
            nn.LeakyReLU(),
            nn.Linear(D_k, D_o),
            nn.Sigmoid(),
        )
        def weights_init(layer_in):
            if isinstance(layer_in, nn.Linear):
                nn.init.kaiming_normal_(layer_in.weight)
                layer_in.bias.data.fill_(0.0)
        self.ffn.apply(weights_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x)






