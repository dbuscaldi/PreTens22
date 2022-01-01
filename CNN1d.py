import torch
from torch import nn
from torch.nn import *
import torch.nn.functional as F

class CNN1d(nn.Module):
    def __init__(self, langmodel, hidden_size, n_filters, filter_sizes):

        super().__init__()

        self.model=langmodel

        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = 768,
                                              out_channels = n_filters,
                                              kernel_size = fs)
                                    for fs in filter_sizes
                                    ])

        self.fc = nn.Linear(len(filter_sizes) * n_filters, hidden_size)

        self.fc1 = nn.Linear(hidden_size, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, text):

        embedded = self.model(input_ids=text)

        #embedded[0] size: [batch size, sent len, emb dim] -> switching sent len and emb dim

        embedded = embedded[0].permute(0, 2, 1)

        #embedded = [batch size, emb dim, sent len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]

        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]

        #pooled_n = [batch size, n_filters]

        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]

        out = F.relu(self.fc(cat))

        return self.fc1(out)
