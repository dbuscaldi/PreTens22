import torch
from torch import nn
from torch.nn import *
import torch.nn.functional as F

class biLSTM(nn.Module):
    def __init__(self, langmodel, hidden_size=50, dropout=0.5):
        super(biLSTM, self).__init__()
        self.dropout = dropout
        self.model=langmodel
        self.hidden_size=hidden_size
        self.lstm = nn.LSTM(input_size=768, hidden_size=self.hidden_size, bidirectional=True, batch_first=True)
        self.dense1 = nn.Linear(self.hidden_size *2, self.hidden_size * 2)
        self.dense = nn.Linear(self.hidden_size * 2, 1)
        #self.sig = nn.Sigmoid()

    def forward(self, x, hidden):
        #print("x",x.shape)
        current_batch_size=x.shape[0]
        outputs = self.model(input_ids=x)
        output, (hidden_h, hidden_c) = self.lstm(outputs[0], hidden)

        output_hidden = torch.cat((hidden_h[0], hidden_h[1]), dim=1)
        output_hidden=self.dense1(F.dropout(output_hidden, self.dropout))
        out = F.relu(output_hidden)
        out = self.dense(F.dropout(out, self.dropout))
        #sig_out = self.sig(out).view(current_batch_size, -1)

        #sig_out = sig_out[:, -1] # get last batch of labels
        out = out.view(current_batch_size, -1)
        hidden = (hidden_h, hidden_c)

        return out, hidden

    def init_bilstm_hidden(self, batch_size, device):
        h0 = torch.zeros(2, batch_size, self.hidden_size).to(device) # 2 for bidirection
        c0 = torch.zeros(2, batch_size, self.hidden_size).to(device)

        return (h0, c0)
