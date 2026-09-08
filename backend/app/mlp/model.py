import torch
import torch.nn as nn

class DeeperMLP(nn.Module):
    def __init__(self, input_dim=13, num_classes=2):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.fc_out = nn.Linear(32, num_classes)

    def forward(self, x):
        h1 = self.relu1(self.fc1(x))
        h2 = self.relu2(self.fc2(h1))
        out = self.fc_out(h2)
        return out
