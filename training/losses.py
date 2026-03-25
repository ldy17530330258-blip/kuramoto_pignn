import torch

from configs import kuramoto_config as config
from physics.kuramoto_physics import (
    angle_supervision_loss,
    kuramoto_residual,
    compute_order_parameter,
)


def batch_order_parameter(theta_pred: torch.Tensor, batch_vec: torch.Tensor | None):
    R, _ = compute_order_parameter(theta_pred, batch_vec)
    return R


def compute_data_loss(theta_pred_next: torch.Tensor, theta_true_next: torch.Tensor):
    return angle_supervision_loss(theta_pred_next, theta_true_next)


def _expand_graph_scalar_to_nodes(
    graph_value: torch.Tensor,
    batch_vec: torch.Tensor | None,
    num_nodes: int
):
    """
    graph_value:
        single graph => shape [1] or scalar
        batched graphs => shape [B] or [B, 1]
    return:
        node-wise tensor [N]
    """
    gv = graph_value.view(-1)
    if batch_vec is None:
        if gv.numel() == 1:
            return gv.expand(num_nodes)
        raise ValueError(
            f"Expected scalar graph value for single graph, got shape {tuple(graph_value.shape)}"
        )
    return gv[batch_vec]


def compute_physics_loss(delta_theta: torch.Tensor, batch):
    batch_vec = getattr(batch, 'batch', None)
    K_per_node = _expand_graph_scalar_to_nodes(batch.K, batch_vec, batch.theta_t.numel())
    dt_val = float(batch.dt.view(-1)[0].item()) if hasattr(batch, 'dt') else float(config.DT)

    res = kuramoto_residual(
        delta_theta_pred=delta_theta,
        theta_t=batch.theta_t,
        omega=batch.omega,
        edge_index=batch.edge_index,
        coupling_K=K_per_node,
        dt=dt_val,
        alive_mask=batch.alive_mask,
    )
    return res.pow(2).mean()


def compute_ic_loss(theta_pred_next: torch.Tensor, batch):
    """
    当前 v1 是 one-step supervised rollout：
    输入里已经显式给定 theta_t（通过 sin(theta_t), cos(theta_t)）。
    模型学的是 t -> t+dt，而不是直接学习整条 theta(t) 轨迹函数。

    因此这里不存在一个需要额外优化的独立 IC（初始条件）损失。
    之前那种 time_id == 0 时再对首步样本加一次监督，
    本质上只是重复加权首步 data loss，不是标准 IC 约束。
    """
    return theta_pred_next.new_zeros(())


def compute_r_loss(theta_pred_next: torch.Tensor, batch):
    batch_vec = getattr(batch, 'batch', None)
    R_pred = batch_order_parameter(theta_pred_next, batch_vec)
    R_true = batch.R_next.view(-1)
    return (R_pred - R_true).pow(2).mean()


def compute_total_loss(model_out, batch):
    theta_pred_next = model_out['theta_pred_next']
    delta_theta = model_out['delta_theta']

    l_data = compute_data_loss(theta_pred_next, batch.theta_next)
    l_phy = compute_physics_loss(delta_theta, batch)
    l_ic = compute_ic_loss(theta_pred_next, batch)
    l_R = compute_r_loss(theta_pred_next, batch)

    total = (
        l_data
        + config.LAMBDA_PHY * l_phy
        + config.LAMBDA_IC * l_ic
        + config.LAMBDA_R * l_R
    )

    return total, {
        'total': float(total.detach().item()),
        'L_data': float(l_data.detach().item()),
        'L_phy': float(l_phy.detach().item()),
        'L_ic': float(l_ic.detach().item()),
        'L_R': float(l_R.detach().item()),
    }