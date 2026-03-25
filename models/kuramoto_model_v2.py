import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StaticNodeEncoder(nn.Module):
    """
    只编码静态/慢变量，不把 R_t / Psi_t 这种宏观先验喂进去，
    也不把现成的 sin(theta_i), cos(theta_i) 直接送进静态编码器。
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        # x_static = [omega_i, deg_i, clust_i, alive_mask_i]
        self.mlp = MLP(in_dim=4, hidden_dim=hidden_dim, out_dim=hidden_dim, dropout=dropout)

    def forward(self, x_static: torch.Tensor) -> torch.Tensor:
        return self.mlp(x_static)


class EdgeMessageBlock(nn.Module):
    """
    Physics-aligned edge message:
    msg_{j->i} = MLP(h_i, h_j, sin(theta_j-theta_i), cos(theta_j-theta_i), K)
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.edge_mlp = MLP(
            in_dim=2 * hidden_dim + 3,   # h_i, h_j, sinΔ, cosΔ, K
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            dropout=dropout,
        )

    def forward(
            self,
            h: torch.Tensor,
            theta_t: torch.Tensor,
            edge_index: torch.Tensor,
            K_per_node: torch.Tensor,
            alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        h         : [N, H]
        theta_t   : [N]
        edge_index: [2, E]  (src=j, dst=i)
        K_per_node: [N]
        alive_mask: [N]
        return    : aggregated edge message per node, [N, H]
        """
        src, dst = edge_index  # j -> i
        target_dtype = h.dtype

        theta_src = theta_t[src]
        theta_dst = theta_t[dst]

        sin_delta = torch.sin(theta_src - theta_dst).unsqueeze(-1).to(target_dtype)  # [E, 1]
        cos_delta = torch.cos(theta_src - theta_dst).unsqueeze(-1).to(target_dtype)  # [E, 1]

        h_src = h[src].to(target_dtype)  # [E, H]
        h_dst = h[dst].to(target_dtype)  # [E, H]

        K_edge = K_per_node[dst].unsqueeze(-1).to(target_dtype)  # [E, 1]

        edge_feat = torch.cat([h_dst, h_src, sin_delta, cos_delta, K_edge], dim=-1).to(target_dtype)
        msg = self.edge_mlp(edge_feat)  # [E, H]

        edge_alive = (alive_mask[src] * alive_mask[dst]).unsqueeze(-1).to(msg.dtype)  # [E, 1]
        msg = msg * edge_alive

        N = h.size(0)
        agg = torch.zeros(N, msg.size(-1), device=h.device, dtype=msg.dtype)
        agg.index_add_(0, dst, msg)  # sum over incoming edges

        return agg


class NodeUpdateDecoder(nn.Module):
    """
    Δtheta_i = MLP(h_i, m_i, omega_i, sin(theta_i), cos(theta_i))
    """
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.node_mlp = MLP(
            in_dim=2 * hidden_dim + 3,   # h_i, m_i, omega_i, sin(theta_i), cos(theta_i)
            hidden_dim=hidden_dim,
            out_dim=1,
            dropout=dropout,
        )

    def forward(
            self,
            h: torch.Tensor,
            agg_msg: torch.Tensor,
            omega: torch.Tensor,
            theta_t: torch.Tensor,
            alive_mask: torch.Tensor,
    ) -> torch.Tensor:
        target_dtype = h.dtype

        agg_msg = agg_msg.to(target_dtype)
        omega = omega.unsqueeze(-1).to(target_dtype)
        sin_theta = torch.sin(theta_t).unsqueeze(-1).to(target_dtype)
        cos_theta = torch.cos(theta_t).unsqueeze(-1).to(target_dtype)

        node_feat = torch.cat([h, agg_msg, omega, sin_theta, cos_theta], dim=-1)
        delta_theta = self.node_mlp(node_feat).squeeze(-1)

        delta_theta = delta_theta * alive_mask.to(delta_theta.dtype)
        return delta_theta


class KuramotoPIGNN_V2(nn.Module):
    """
    v2: Edge-Message Physics-Informed GNN

    兼容当前 trainer / losses / rollout_eval 的输出接口：
    {
        'delta_theta': ...,
        'theta_pred_next': ...
    }
    """
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        # 为了兼容现有 trainer 签名，保留 input_dim / num_layers 形参
        # 但 v2 的核心不是 GCN 层数，而是 edge-message 结构
        self.hidden_dim = hidden_dim
        self.static_encoder = StaticNodeEncoder(hidden_dim=hidden_dim, dropout=dropout)
        self.edge_block = EdgeMessageBlock(hidden_dim=hidden_dim, dropout=dropout)
        self.decoder = NodeUpdateDecoder(hidden_dim=hidden_dim, dropout=dropout)

        # 一个轻量残差投影，让静态隐变量更稳
        self.post_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _expand_graph_scalar_to_nodes(
        graph_value: torch.Tensor,
        batch_vec: torch.Tensor | None,
        num_nodes: int,
    ) -> torch.Tensor:
        gv = graph_value.view(-1)
        if batch_vec is None:
            if gv.numel() != 1:
                raise ValueError(f"Expected single graph scalar, got shape {tuple(graph_value.shape)}")
            return gv.expand(num_nodes)
        return gv[batch_vec]

    def forward(self, data):
        """
        当前数据集的 x 列顺序：
        [omega, deg, clust, alive_mask, sin(theta), cos(theta), K, R_t, sin(Psi_t), cos(Psi_t)]

        v2 有意不直接使用 R_t / Psi_t / K 作为节点主干输入，
        只用静态项做 node embedding，再在 edge-message 里显式注入 K 和相位差。
        """
        x = data.x
        theta_t = data.theta_t
        omega = data.omega
        alive_mask = data.alive_mask
        edge_index = data.edge_index
        batch = getattr(data, 'batch', None)

        # 只保留静态/慢变量: [omega, deg, clust, alive_mask]
        x_static = x[:, :4]  # [N, 4]
        h = self.static_encoder(x_static)
        h = h + self.post_proj(h)

        K_per_node = self._expand_graph_scalar_to_nodes(
            graph_value=data.K,
            batch_vec=batch,
            num_nodes=theta_t.numel(),
        )

        agg_msg = self.edge_block(
            h=h,
            theta_t=theta_t,
            edge_index=edge_index,
            K_per_node=K_per_node,
            alive_mask=alive_mask,
        )

        delta_theta = self.decoder(
            h=h,
            agg_msg=agg_msg,
            omega=omega,
            theta_t=theta_t,
            alive_mask=alive_mask,
        )

        theta_pred_next = torch.atan2(
            torch.sin(theta_t + delta_theta),
            torch.cos(theta_t + delta_theta),
        )

        return {
            'delta_theta': delta_theta,
            'theta_pred_next': theta_pred_next,
        }