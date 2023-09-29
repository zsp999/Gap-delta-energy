import torch.nn as nn
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
class MLP(nn.Module):
    def __init__(self,input_num, hidden_layers,dropout,units):
        super(MLP, self).__init__()
        self.input_num = input_num
        self.hidden_layers = hidden_layers
        self.p = dropout
        self.units = units
        self.hidden0 = nn.Sequential(
            nn.Linear(self.input_num, self.units),
            nn.Dropout(p=self.p),
            nn.ReLU()
        )
        self.hidden = nn.Sequential(
            nn.Linear(self.units, self.units),
            nn.Dropout(p=self.p),
            nn.ReLU()
        )

        self.fc = nn.Linear(self.units, 2)


    def forward(self, x):
        x = self.hidden0(x)
        for _ in range(self.hidden_layers):
            x = self.hidden(x)
        x = self.fc(x)

        return x