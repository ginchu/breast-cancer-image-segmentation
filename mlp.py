import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, num_neuron, nonlinear=F.relu):
        super(MLP, self).__init__()

        self.fc1 = nn.Linear(input_dim, num_neuron)
        self.fc2 = nn.Linear(num_neuron, output_dim)
        self.nonlinear = nonlinear    

    def forward(self, x):
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.softmax(x, dim=1)

