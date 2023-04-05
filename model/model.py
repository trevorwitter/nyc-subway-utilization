import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable

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
    def __init__(self, window, num_features, hidden_units, embedding_dim=64, num_layers=1,dropout=0):
        super(Encoder, self).__init__()
        self.window = window
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
        #x = x.reshape((1, self.window, self.num_features))
        #h_1 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_units))
        
        h0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_())
        c0 = Variable(torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_())
        x, (h, c) = self.lstm(x, (h0, c0))
        #print(f"encoder x shape: {x.shape}")
        #print(f"encoder h shape: {h.shape}")
        #print(f"encoder c shape: {c.shape}")
        return x, (h, c)

    
class Decoder(nn.Module):
    def __init__(self, seq_len, input_dim, num_features, dropout=0):
        super(Decoder, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.num_features = num_features
        self.dropout=dropout
        self.lstm = nn.LSTM(
            input_size=self.num_features, 
            hidden_size=input_dim,
            num_layers=1,
            batch_first=True,
            dropout=self.dropout
        )
        self.linear = nn.Linear(self.hidden_dim, self.num_features)

    def forward(self, x, input_h, input_c):
        batch_size = x.shape[0]
        x = x.reshape((batch_size,1,self.num_features))
        x, (hn, cn) = self.lstm(x, (input_h, input_c))
        x = self.linear(x)
        return x, hn, cn

class Attention(nn.Module):
    def __init__(self, enc_hid_dim, dec_hid_dim):
        super().__init__()
        self.attn = nn.Linear((enc_hid_dim) + dec_hid_dim, dec_hid_dim)
        self.v = nn.Linear(dec_hid_dim, 1, bias= False)
    
    def forward(self, hidden, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        src_len = encoder_outputs.shape[1]
        print(hidden.shape)
        hidden = hidden[2:3, :, :]
        hidden = hidden.repeat(1, src_len, 1)
        print(f"attention hidden shape: {hidden.shape}")
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        attention = self.v(energy).squeeze(2)
        return F.softmax(attention, dim=1)
    
class AttentionDecoder(nn.Module):
    def __init__(self, seq_len, attention, input_dim, num_features, encoder_hidden_state = 512, dropout=0):
        super(AttentionDecoder, self).__init__()
        self.seq_len = seq_len
        self.attention = attention
        self.input_dim = input_dim
        self.hidden_dim = input_dim
        self.num_features = num_features
        self.dropout=dropout
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_state + 1, 
            hidden_size=input_dim,
            num_layers=3,
            batch_first=True,
            dropout=self.dropout
        )
        self.linear = nn.Linear(self.hidden_dim * 2, self.num_features)

    def forward(self, x, input_h, input_c, encoder_outputs):
        a = self.attention(input_h, encoder_outputs)
        a = a.unsqueeze(1)
        weighted = torch.bmm(a, encoder_outputs)
        x = x.reshape((1,1,1))
        rnn_input = torch.cat((x, weighted), dim = 2)
        x, (hn, cn) = self.lstm(rnn_input, (input_h, input_c))
        output = x.squeeze(0)
        weighted = weighted.squeeze(0)
        x = self.linear(torch.cat((output, weighted), dim=1))
        return x, hn, cn

class Seq2Seq(nn.Module):
    def __init__(self, num_features, window, horizon, hidden_units, num_layers=1, dropout=0):
        super(Seq2Seq, self).__init__()
        self.num_features = num_features
        self.window = window
        self.horizon = horizon
        self.hidden_units = hidden_units
        self.num_layers = num_layers
        self.dropout = dropout
        #self.attention = Attention(64,64)
        self.encoder = Encoder(
            window=self.window,
            num_features=self.num_features,
            hidden_units=self.hidden_units, 
            num_layers=self.num_layers, 
            dropout=self.dropout
            )
        #self.decoder = AttentionDecoder(
        #    seq_len=window,
        #    attention=self.attention,
        #    input_dim=64,
        #    num_features=self.num_features,
        #    
        #    dropout=self.dropout
        #    #input is output state of encoder
        #    #output length is equal to specified horizon length
        #    )
        self.decoder = Decoder(
            seq_len=self.window, 
            input_dim=self.hidden_units, 
            num_features=self.num_features
            )
    def forward(self,x):
        batch_size = x.shape[0]
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        #o,(h,c) = self.encoder(x, (h0, c0))
        o,(h,c) = self.encoder(x)
        targets_ta = []
        prev_output = x[:,-1,:].unsqueeze(1)
        for horizon_step in range(self.horizon):
            prev_x, prev_h, prev_c = self.decoder(prev_output,h, c)
            targets_ta.append(prev_x.reshape((batch_size, 1, self.num_features)))
            h, c = prev_h, prev_c
            prev_output = prev_x
        targets = torch.stack(targets_ta).squeeze(2)
        targets = targets.reshape(batch_size,self.horizon, self.num_features)
        return targets