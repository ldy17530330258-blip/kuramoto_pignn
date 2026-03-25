import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class NodeEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, dropout: float = 0.1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = conv(h, edge_index)
            h = bn(h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)
        return h


class GlobalContextFusion(nn.Module):
    def __init__(self, hidden_dim: int, global_dim: int = 4):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim + global_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, h: torch.Tensor, global_ctx: torch.Tensor, batch: torch.Tensor | None):
        """
        h         : [N, H]
        global_ctx: single-graph => [4] or [1, 4]
                    batched-graph => [B, 4] (preferred) or flattened [B*4]
        batch     : [N] node-to-graph mapping or None
        """
        if batch is None:
            if global_ctx.dim() == 1:
                global_ctx = global_ctx.view(1, -1)
            elif global_ctx.dim() == 2:
                pass
            else:
                raise ValueError(f"Unexpected global_ctx shape for single graph: {tuple(global_ctx.shape)}")
            g = global_ctx.expand(h.size(0), -1)
        else:
            num_graphs = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

            if global_ctx.dim() == 1:
                # flattened form from PyG collation: [B*4]
                if global_ctx.numel() % num_graphs != 0:
                    raise ValueError(
                        f"Cannot reshape global_ctx of shape {tuple(global_ctx.shape)} into [{num_graphs}, -1]"
                    )
                global_ctx = global_ctx.view(num_graphs, -1)
            elif global_ctx.dim() == 2 and global_ctx.size(0) != num_graphs:
                # defensive reshape if collation produced [1, B*4] or similar
                if global_ctx.numel() % num_graphs != 0:
                    raise ValueError(
                        f"Cannot reshape global_ctx of shape {tuple(global_ctx.shape)} into [{num_graphs}, -1]"
                    )
                global_ctx = global_ctx.view(num_graphs, -1)

            g = global_ctx[batch]  # [N, global_dim]

        return h + self.proj(torch.cat([h, g], dim=-1))


class StateDecoder(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, h):
        return self.mlp(h).squeeze(-1)


class KuramotoPIGNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.encoder = NodeEncoder(input_dim, hidden_dim, num_layers, dropout)
        self.global_fusion = GlobalContextFusion(hidden_dim, global_dim=4)
        self.decoder = StateDecoder(hidden_dim)

    def forward(self, data):
        h = self.encoder(data.x, data.edge_index)
        batch = getattr(data, 'batch', None)
        global_ctx = data.global_ctx
        h = self.global_fusion(h, global_ctx, batch)
        delta_theta = self.decoder(h)
        theta_pred_next = torch.atan2(
            torch.sin(data.theta_t + delta_theta),
            torch.cos(data.theta_t + delta_theta)
        )
        return {
            'delta_theta': delta_theta,
            'theta_pred_next': theta_pred_next,
        }
