import torch
from torch import nn
import torch.nn.functional as F

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
        self.linear = nn.Linear(in_features=self.hidden_units, out_features=self.num_features)
        
    def forward(self,x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        _,(hn,_) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0])#.flatten()
        return out


class Encoder(nn.Module):
    def __init__(self, num_features, hidden_units, num_layers=1,dropout=0):
        super(Encoder, self).__init__()
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
            
    def forward(self,x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        x, (h, c) = self.lstm(x, (h0, c0))
        return x, h, c


class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias= False)
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        hidden = hidden[2:3, :, :]
        hidden = hidden.repeat(1, src_len, 1)
        energy = torch. tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)

class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, num_features, dropout=0):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.num_features = num_features
        self.dropout=dropout
        self.lstm = nn.LSTM(
            input_size=1, 
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=self.dropout
        )
        self.linear = nn.Linear(self.hidden_dim, self.num_features)

    def forward(self, x, input_h, input_c):
        x = x.reshape((1,1,1))
        x, (hn, cn) = self.lstm(x, (input_h, input_c))
        x = self.linear(x)
        return x, hn, cn

class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim, num_features, dropout=0):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.num_features = num_features
        self.attention = attention
        self.dropout=dropout

        self.lstm = nn.LSTM(
            input_size=encoder_hidden_state + 1, 
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=self.dropout
        )
        self.linear = nn.Linear(self.hidden_dim*2, self.num_features)

    def forward(self, x, input_h, input_c):
        x = x.reshape((1,1,1))
        x, (hn, cn) = self.lstm(x, (input_h, input_c))
        x = self.linear(x)
        return x, hn, cn

class Seq2Seq(nn.Module):
    def __init__(self, num_features, horizon, encoder_hidden_units, encoder_num_layers=1, dropout=0):
        super().__init__()
        self.num_features = num_features
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        self.encoder = Encoder(
            num_features=self.num_features,
            hidden_units=self.hidden_units, 
            num_layers=self.num_layers, 
            dropout=self.dropout
            )
        self.decoder = nn.LSTM(
            #input is output state of encoder
            #output length is equal to specified horizon length
            )
    
    def forward(self,x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        o,(hn,_) = self.encoder(x, (h0, c0))
        
        out = self.decoder() #decoder takes output of encoder
        return out