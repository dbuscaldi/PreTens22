import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, SAGPooling
from torch_geometric.nn.glob import global_max_pool

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, 32, num_relations)
        self.conv2 = RGCNConv(32, 16, num_relations)
        self.dense = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_max_pool(x, data.batch)
        x = self.dense(x)
        
        return x
        #return F.log_softmax(x, dim=1)
