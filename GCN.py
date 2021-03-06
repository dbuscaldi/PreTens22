import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv, FastRGCNConv, FiLMConv, GATConv, SAGPooling
from torch_geometric.nn.glob import global_max_pool, global_mean_pool

class RGCN(torch.nn.Module):
    def __init__(self, num_node_features, num_relations):
        super().__init__()
        self.conv1 = RGCNConv(num_node_features, 32, num_relations)
        #self.conv1 = FiLMConv(num_node_features, 32, num_relations) #FiLMConv looks like a better choice than RGCNConv but it bugs on GPU
        self.conv2 = RGCNConv(32, 16, num_relations)
        #self.conv2 = FiLMConv(32, 16, num_relations)
        self.dense = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = global_mean_pool(x, data.batch) #note: looks like mean pooling is working better than max pooling
        x = self.dense(x)

        return x
        #return F.log_softmax(x, dim=1)

class GAT(torch.nn.Module):
    def __init__(self, num_node_features):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 32, heads=4, add_self_loops=False, edge_dim=1)
        self.dense1 = nn.Linear(32*4, 32)
        self.dense2 = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        (x, edge_attns) = self.conv1(x, edge_index, return_attention_weights=True)
        #we may use the attentions later for analysing the results, for now we discard them

        #print(data.num_nodes, x.shape) #num nodes, (h_size * N_heads)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = global_max_pool(x, data.batch)
        x = self.dense2(x)
        #print(x)

        return x

class Parallel(torch.nn.Module):
    #takes both RGCN and GATConv in parallel
    def __init__(self, num_node_features, num_relations):
        super().__init__()
        self.conv1 = GATConv(num_node_features, 32, heads=4, add_self_loops=False, edge_dim=1)
        self.dense1 = nn.Linear(32*4, 16)

        self.rconv1 = RGCNConv(num_node_features, 32, num_relations)
        self.rconv2 = RGCNConv(32, 16, num_relations)
        self.dense = nn.Linear(32, 1)

    def forward(self, data):
        in_x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        x = self.conv1(in_x, edge_index)

        #print(data.num_nodes, x.shape) #num nodes, (h_size * N_heads)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = global_max_pool(x, data.batch)

        z = self.rconv1(in_x, edge_index, edge_attr)
        z = F.relu(z)
        z = F.dropout(z, 0.5)
        z = self.rconv2(z, edge_index, edge_attr)
        z = F.relu(z)
        z = global_mean_pool(z, data.batch)

        z = torch.cat((x,z),1)
        out = self.dense(z)
        #print(x)

        return out

class Mixed(torch.nn.Module):
    #takes both RGCN and GATConv in parallel
    def __init__(self, num_node_features, num_relations):
        super().__init__()
        self.rconv1 = RGCNConv(num_node_features, 64, num_relations)
        self.conv1 = GATConv(64, 32, heads=4, add_self_loops=False, edge_dim=1)
        self.dense1 = nn.Linear(32*4, 16)
        self.dense2 = nn.Linear(16, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, (data.edge_attr).flatten()

        x = self.rconv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, 0.5)
        x = self.conv1(x, edge_index)

        #print(data.num_nodes, x.shape) #num nodes, (h_size * N_heads)
        x = F.leaky_relu(x)
        x = F.dropout(x, 0.5)
        x = self.dense1(x)
        x = F.leaky_relu(x)
        x = global_max_pool(x, data.batch)

        x = self.dense2(x)
        #print(x)

        return x
