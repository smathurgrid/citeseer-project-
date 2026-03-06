import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class GCN(torch.nn.Module): # this means this is a neural network 
    def __init__(self, in_channels, hidden_channels, out_channels, dropout=0.5):
        super(GCN, self).__init__()
        # what layers I have created 

        self.conv1 = GCNConv(in_channels, hidden_channels)
        # in this input is going 3703 features per paper 
        # Output = hiddle size 
        self.conv2 = GCNConv(hidden_channels, out_channels)
        # Input is given as hidden size 
        # Output is given as 6(Number of class)
        self.dropout = dropout
        # So basically the structure is 
        # 3703 features -> Hidden -> 6 

    def forward(self, x, edge_index):

        x = self.conv1(x, edge_index)
        # Forward Pass 
        # Each paper looks at its neighbors 
        # Mixes its features with neighbor features 
        # Produces new representation 
        x = F.relu(x)
        # Apply activation (non-linearity)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        # randomly droping some neurons (for regularization)
        x = self.conv2(x, edge_index)
        # Look at neighbors again
        # Propagate information 
        # Produce final 6 scores 


        return F.log_softmax(x, dim=1)
        # Convert scores into probablities (log prob)
# SUMMARY OF GCN 
# Looks at its own words 
# Looks at neighbor papers 
# Learns from neighbors
# Predicts topic
# Two ronuds of propagation         
from torch_geometric.nn import APPNP
import torch.nn.functional as F
import torch

class APPNPModel(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 K=10, alpha=0.3, dropout=0.7):
        super(APPNPModel, self).__init__()
    

        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
    #Fully connected layer 
        self.prop = APPNP(K=K, alpha=alpha)
    # This is the propagation layer 
        self.dropout = dropout

    def forward(self, x, edge_index):
        # Standard neural network on features.
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)
    # Now propagation happens.
        x = self.prop(x, edge_index)
    # connvert to probabalities 
        return F.log_softmax(x, dim=1)
    

    ## MAJOR DIFF BETWEEN APPNP AND GCN 
    # GCN - Propagation is inside each layer 
    #       usually 2 layers -> 2 propagation steps
    # APPNP - First compute features.
    #         Then propagate many times (K times)
    # Soo APPNP can spread information farther.



# ===========================
# GPR-GNN (Stable Implementation)
# ===========================

from torch_geometric.nn import APPNP

class GPRGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 K=10, alpha=0.1, dropout=0.5):
        super(GPRGNN, self).__init__()

        self.lin1 = torch.nn.Linear(in_channels, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        self.prop = APPNP(K=1, alpha=alpha)  
        # We will manually apply propagation K times

        self.gamma = torch.nn.Parameter(torch.Tensor(K + 1))
        torch.nn.init.uniform_(self.gamma)

        self.K = K
        self.dropout = dropout

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin2(x)

        hidden = x
        out = self.gamma[0] * hidden

        for k in range(1, self.K + 1):
            hidden = self.prop(hidden, edge_index)
            out += self.gamma[k] * hidden

        return F.log_softmax(out, dim=1)
