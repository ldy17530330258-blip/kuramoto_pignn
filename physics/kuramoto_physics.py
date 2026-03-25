import torch


def wrap_angle(theta: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(theta), torch.cos(theta))


def compute_order_parameter(theta: torch.Tensor, batch: torch.Tensor | None = None):
    """
    theta: [N]
    batch: [N] or None
    returns:
        R   [B]
        psi [B]
    """
    if batch is None:
        z = torch.exp(1j * theta).mean()
        return torch.abs(z).view(1), torch.angle(z).view(1)

    B = int(batch.max().item()) + 1 if batch.numel() > 0 else 1
    R_list, psi_list = [], []
    for b in range(B):
        mask = (batch == b)
        z = torch.exp(1j * theta[mask]).mean()
        R_list.append(torch.abs(z))
        psi_list.append(torch.angle(z))
    return torch.stack(R_list), torch.stack(psi_list)


def kuramoto_rhs(
    theta: torch.Tensor,
    omega: torch.Tensor,
    edge_index: torch.Tensor,
    coupling_K: torch.Tensor,
    alive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    theta      : [N]
    omega      : [N]
    edge_index : [2, E]
    coupling_K : scalar or [N] node-wise K
    alive_mask : [N] or None
    """
    src, dst = edge_index
    if alive_mask is None:
        alive_mask = torch.ones_like(theta)

    edge_term = torch.sin(theta[src] - theta[dst]) * alive_mask[src] * alive_mask[dst]
    agg = torch.zeros_like(theta).index_add(0, dst, edge_term)

    rhs = omega.clone()
    rhs = rhs + coupling_K * agg
    rhs = rhs * alive_mask
    return rhs


def kuramoto_residual(
    delta_theta_pred: torch.Tensor,
    theta_t: torch.Tensor,
    omega: torch.Tensor,
    edge_index: torch.Tensor,
    coupling_K: torch.Tensor,
    dt: float,
    alive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    rhs = kuramoto_rhs(theta_t, omega, edge_index, coupling_K, alive_mask)
    return delta_theta_pred / dt - rhs


def angle_supervision_loss(theta_pred_next: torch.Tensor, theta_true_next: torch.Tensor) -> torch.Tensor:
    return (
        (torch.sin(theta_pred_next) - torch.sin(theta_true_next)) ** 2
        + (torch.cos(theta_pred_next) - torch.cos(theta_true_next)) ** 2
    ).mean()
