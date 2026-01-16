import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

def get_activation_fn(name):
    if name == 'ReLU': return nn.ReLU()
    elif name == 'Tanh': return nn.Tanh()
    elif name == 'LeakyReLU': return nn.LeakyReLU()
    elif name == 'ELU': return nn.ELU()
    else: raise ValueError(f"Unknown activation function: {name}")

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        probs = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        loss = self.alpha * focal_weight * bce_loss
        return loss.mean() if self.reduction == 'mean' else loss.sum()

class NodeMLP_GCN(nn.Module):
    def __init__(self, in_node_feats, hidden_dim=256, num_gcn_layers=5,
                 dropout=0.3, use_residual=True, use_batch_norm=False,
                 activation='ReLU', use_bias=True):
        super().__init__()
        self.use_residual = use_residual
        self.use_batch_norm = use_batch_norm
        self.activation_fn = get_activation_fn(activation)

        # Updated node MLP structure as per request
        self.node_mlp = nn.Sequential(
            nn.Linear(in_node_feats, 64, bias=use_bias),
            self.activation_fn,
            nn.Linear(64, 128, bias=use_bias),
            self.activation_fn,
            nn.Linear(128, 128, bias=use_bias),
            self.activation_fn,
            nn.Linear(128, 256, bias=use_bias),
            self.activation_fn,
            nn.Linear(256, hidden_dim, bias=use_bias),  # final projection to hidden_dim
            self.activation_fn,
            nn.Dropout(dropout)
        )

        if use_batch_norm:
            self.bn_layers = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_gcn_layers)])
        else:
            self.bn_layers = None

        self.gcn_layers = nn.ModuleList(
            [GCNConv(hidden_dim, hidden_dim, bias=use_bias) for _ in range(num_gcn_layers)]
        )

        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.node_mlp(x)
        for i, gcn in enumerate(self.gcn_layers):
            x_res = x
            x = gcn(x, edge_index)
            if self.bn_layers:
                x = self.bn_layers[i](x)
            x = self.activation_fn(x)
            if self.use_residual:
                x = x + x_res
        logits = self.out(x).squeeze(-1)
        probs = torch.sigmoid(logits)
        return logits, probs

