import torch
from torch import nn


class LSTMRegression(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers=1,dropout=0):
        super().__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        
    def forward(self,x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        _,(hn,_) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()
        return out