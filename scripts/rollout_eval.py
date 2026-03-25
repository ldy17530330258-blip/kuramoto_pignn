import os
import sys
import json
import math
import pickle
import argparse
import random

import numpy as np
import torch
import matplotlib.pyplot as plt
from torch_geometric.data import Data

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == 'scripts' else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import kuramoto_config as config
from models.kuramoto_model import KuramotoPIGNN
from models.kuramoto_model_v2 import KuramotoPIGNN_V2
from physics.kuramoto_physics import compute_order_parameter


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('[Warn] CUDA unavailable, fallback to CPU.')
        return torch.device('cpu')
    return config.DEVICE


def use_v2_arch(tag: str) -> bool:
    return tag == 'v2_edge' or tag.startswith('v2_')


def build_model_by_tag(tag: str, device: torch.device):
    if use_v2_arch(tag):
        model = KuramotoPIGNN_V2(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_GNN_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)
    else:
        model = KuramotoPIGNN(
            input_dim=config.NODE_FEATURE_DIM,
            hidden_dim=config.HIDDEN_DIM,
            num_layers=config.NUM_GNN_LAYERS,
            dropout=config.DROPOUT,
        ).to(device)
    return model


def load_dataset_and_split():
    data_path = os.path.join(config.PYG_DIR, 'kuramoto_dataset.pt')
    split_path = os.path.join(config.PYG_DIR, 'graph_split.pkl')
    if not os.path.exists(data_path):
        raise FileNotFoundError(f'Dataset not found: {data_path}')
    if not os.path.exists(split_path):
        raise FileNotFoundError(f'Split file not found: {split_path}')

    data_list = torch.load(data_path, map_location='cpu', weights_only=False)
    with open(split_path, 'rb') as f:
        train_graphs, val_graphs, test_graphs = pickle.load(f)

    split_map = {
        'train': set(train_graphs),
        'val': set(val_graphs),
        'test': set(test_graphs),
    }
    return data_list, split_map


def load_model(tag: str, device: torch.device):
    ckpt_path = os.path.join(config.CKPT_DIR, f'kuramoto_pignn_{tag}_best.pt')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    model = build_model_by_tag(tag, device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    return model, ckpt_path, ckpt


def get_graph_samples(data_list, graph_id: int):
    samples = [d for d in data_list if int(d.graph_id.item()) == graph_id]
    if len(samples) == 0:
        raise ValueError(f'No samples found for graph_id={graph_id}')
    samples = sorted(samples, key=lambda d: int(d.time_id.item()))
    return samples


def select_graph_id(data_list, split_map, split_name: str, graph_id: int | None):
    valid_ids = sorted(split_map[split_name])
    if graph_id is None:
        if len(valid_ids) == 0:
            raise ValueError(f'No graph ids in split={split_name}')
        return int(valid_ids[0])
    if graph_id not in split_map[split_name]:
        raise ValueError(f'graph_id={graph_id} not in split={split_name}. valid example ids: {valid_ids[:10]}')
    return int(graph_id)


def compute_order_parameter_np(theta: np.ndarray):
    z = np.exp(1j * theta).mean()
    return float(np.abs(z)), float(np.angle(z))


def circular_abs_error(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    diff = np.arctan2(np.sin(pred - true), np.cos(pred - true))
    return np.abs(diff)


def circular_mae(pred: np.ndarray, true: np.ndarray) -> float:
    return float(circular_abs_error(pred, true).mean())


def make_model_input(theta_t: torch.Tensor,
                     edge_index: torch.Tensor,
                     omega: torch.Tensor,
                     deg: torch.Tensor,
                     clust: torch.Tensor,
                     alive_mask: torch.Tensor,
                     K: float,
                     dt: float,
                     device: torch.device) -> Data:
    R_t, psi_t = compute_order_parameter(theta_t, batch=None)
    R_scalar = float(R_t.view(-1)[0].item())
    psi_scalar = float(psi_t.view(-1)[0].item())

    N = theta_t.numel()
    x = torch.stack([
        omega,
        deg,
        clust,
        alive_mask,
        torch.sin(theta_t),
        torch.cos(theta_t),
        torch.full((N,), K, dtype=torch.float32, device=device),
        torch.full((N,), R_scalar, dtype=torch.float32, device=device),
        torch.full((N,), math.sin(psi_scalar), dtype=torch.float32, device=device),
        torch.full((N,), math.cos(psi_scalar), dtype=torch.float32, device=device),
    ], dim=1)

    data = Data(
        x=x,
        edge_index=edge_index,
        theta_t=theta_t,
        omega=omega,
        alive_mask=alive_mask,
        K=torch.tensor([[K]], dtype=torch.float32, device=device),
        dt=torch.tensor([[dt]], dtype=torch.float32, device=device),
        global_ctx=torch.tensor([[K, R_scalar, math.sin(psi_scalar), math.cos(psi_scalar)]],
                                dtype=torch.float32, device=device),
    )
    return data


def rollout_closed_loop(model, samples, device: torch.device, rollout_steps: int):
    first = samples[0]
    edge_index = first.edge_index.to(device)
    omega = first.omega.to(device)
    alive_mask = first.alive_mask.to(device)
    deg = first.x[:, 1].to(device)
    clust = first.x[:, 2].to(device)
    K = float(first.K.view(-1)[0].item())
    dt = float(first.dt.view(-1)[0].item())
    net_type = getattr(first, 'net_type', 'unknown')

    max_available_steps = len(samples)
    rollout_steps = min(int(rollout_steps), max_available_steps)

    theta_true = [samples[0].theta_t.cpu().numpy().astype(np.float32)]
    for s in samples[:rollout_steps]:
        theta_true.append(s.theta_next.cpu().numpy().astype(np.float32))
    theta_true = np.stack(theta_true, axis=0)  # [T+1, N]

    R_true = np.array([compute_order_parameter_np(th)[0] for th in theta_true], dtype=np.float32)
    psi_true = np.array([compute_order_parameter_np(th)[1] for th in theta_true], dtype=np.float32)

    theta_pred_list = []
    theta_curr = samples[0].theta_t.to(device)
    theta_pred_list.append(theta_curr.detach().cpu().numpy().astype(np.float32))

    with torch.no_grad():
        for _ in range(rollout_steps):
            data = make_model_input(
                theta_t=theta_curr,
                edge_index=edge_index,
                omega=omega,
                deg=deg,
                clust=clust,
                alive_mask=alive_mask,
                K=K,
                dt=dt,
                device=device,
            )
            out = model(data)
            theta_curr = out['theta_pred_next']
            theta_pred_list.append(theta_curr.detach().cpu().numpy().astype(np.float32))

    theta_pred = np.stack(theta_pred_list, axis=0)  # [T+1, N]
    R_pred = np.array([compute_order_parameter_np(th)[0] for th in theta_pred], dtype=np.float32)
    psi_pred = np.array([compute_order_parameter_np(th)[1] for th in theta_pred], dtype=np.float32)

    per_step_phase_mae = np.array([
        circular_mae(theta_pred[t], theta_true[t]) for t in range(theta_true.shape[0])
    ], dtype=np.float32)
    per_step_R_abs_err = np.abs(R_pred - R_true).astype(np.float32)

    summary = {
        'num_nodes': int(theta_true.shape[1]),
        'rollout_steps': int(rollout_steps),
        'dt': float(dt),
        'K': float(K),
        'net_type': str(net_type),
        'phase_mae_last': float(per_step_phase_mae[-1]),
        'phase_mae_mean': float(per_step_phase_mae.mean()),
        'R_abs_err_last': float(per_step_R_abs_err[-1]),
        'R_abs_err_mean': float(per_step_R_abs_err.mean()),
    }

    return {
        'theta_true': theta_true,
        'theta_pred': theta_pred,
        'R_true': R_true,
        'R_pred': R_pred,
        'psi_true': psi_true,
        'psi_pred': psi_pred,
        'per_step_phase_mae': per_step_phase_mae,
        'per_step_R_abs_err': per_step_R_abs_err,
        'summary': summary,
    }


def choose_node_indices(num_nodes: int, num_node_plots: int):
    if num_nodes <= num_node_plots:
        return list(range(num_nodes))
    return np.linspace(0, num_nodes - 1, num=num_node_plots, dtype=int).tolist()


def save_plots(result, out_prefix: str, num_node_plots: int = 6):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    theta_true = result['theta_true']
    theta_pred = result['theta_pred']
    R_true = result['R_true']
    R_pred = result['R_pred']
    phase_mae = result['per_step_phase_mae']
    R_err = result['per_step_R_abs_err']

    T_plus_1, N = theta_true.shape
    steps = np.arange(T_plus_1)
    node_ids = choose_node_indices(N, num_node_plots)

    # R(t) curve
    plt.figure(figsize=(7, 4))
    plt.plot(steps, R_true, label='True R(t)')
    plt.plot(steps, R_pred, '--', label='Pred R(t)')
    plt.xlabel('rollout step')
    plt.ylabel('R(t)')
    plt.title('Kuramoto rollout: order parameter')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + '_R_curve.png', dpi=180)
    plt.close()

    # phase error vs horizon
    plt.figure(figsize=(7, 4))
    plt.plot(steps, phase_mae, label='phase circular MAE')
    plt.plot(steps, R_err, label='|R_pred - R_true|')
    plt.xlabel('rollout step')
    plt.ylabel('error')
    plt.title('Rollout error growth')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_prefix + '_error_curve.png', dpi=180)
    plt.close()

    # representative node trajectories (unwrap 后再画，避免 [-pi, pi] 边界视觉假象)
    ncols = 2
    nrows = int(math.ceil(len(node_ids) / ncols))
    plt.figure(figsize=(10, 3.2 * nrows))
    for i, nid in enumerate(node_ids, start=1):
        ax = plt.subplot(nrows, ncols, i)

        theta_true_u = np.unwrap(theta_true[:, nid])
        theta_pred_u = np.unwrap(theta_pred[:, nid])

        ax.plot(steps, theta_true_u, label='true')
        ax.plot(steps, theta_pred_u, '--', label='pred')
        ax.set_title(f'node {nid}')
        ax.set_xlabel('rollout step')
        ax.set_ylabel('theta (unwrapped)')
        ax.grid(alpha=0.3)
        if i == 1:
            ax.legend()

    plt.suptitle('Representative node phase trajectories (unwrapped)')
    plt.tight_layout()
    plt.savefig(out_prefix + '_theta_traj.png', dpi=180)
    plt.close()


def save_summary_json(result, out_prefix: str, graph_id: int, split_name: str, ckpt_path: str):
    out = {
        'graph_id': int(graph_id),
        'split': split_name,
        'checkpoint': ckpt_path,
        'summary': result['summary'],
        'R_true': result['R_true'].tolist(),
        'R_pred': result['R_pred'].tolist(),
        'per_step_phase_mae': result['per_step_phase_mae'].tolist(),
        'per_step_R_abs_err': result['per_step_R_abs_err'].tolist(),
    }
    with open(out_prefix + '_summary.json', 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, ensure_ascii=False)


def parse_args():
    parser = argparse.ArgumentParser(description='Closed-loop rollout evaluation for Kuramoto-PIGNN')
    parser.add_argument('--tag', type=str, default='v1',
                        help='checkpoint tag, e.g. v1 / pure_data / R_guided / v2_edge')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--graph_id', type=int, default=None, help='specific graph id in the chosen split')
    parser.add_argument('--rollout_steps', type=int, default=50, help='closed-loop rollout steps')
    parser.add_argument('--num_node_plots', type=int, default=6, help='number of representative nodes to plot')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default=None, help='override figure/log output directory')
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = pick_device(args.device)

    data_list, split_map = load_dataset_and_split()
    graph_id = select_graph_id(data_list, split_map, args.split, args.graph_id)
    samples = get_graph_samples(data_list, graph_id)

    model, ckpt_path, ckpt = load_model(args.tag, device)

    result = rollout_closed_loop(
        model=model,
        samples=samples,
        device=device,
        rollout_steps=args.rollout_steps,
    )

    base_dir = args.save_dir if args.save_dir is not None else config.FIG_DIR
    os.makedirs(base_dir, exist_ok=True)
    out_prefix = os.path.join(base_dir, f'rollout_{args.tag}_{args.split}_gid{graph_id}')

    save_plots(result, out_prefix=out_prefix, num_node_plots=args.num_node_plots)
    save_summary_json(result, out_prefix=out_prefix, graph_id=graph_id, split_name=args.split, ckpt_path=ckpt_path)

    print('=' * 100)
    print('Closed-loop rollout evaluation finished')
    print(f'device      : {device}')
    print(f'checkpoint  : {ckpt_path}')
    print(f'best_epoch  : {ckpt.get("epoch", "NA")}')
    print(f'graph_id    : {graph_id}')
    print(f'split       : {args.split}')
    print(f'net_type    : {result["summary"]["net_type"]}')
    print(f'num_nodes   : {result["summary"]["num_nodes"]}')
    print(f'rollout     : {result["summary"]["rollout_steps"]} steps')
    print(f'phase_mae_last : {result["summary"]["phase_mae_last"]:.6f}')
    print(f'phase_mae_mean : {result["summary"]["phase_mae_mean"]:.6f}')
    print(f'R_abs_err_last : {result["summary"]["R_abs_err_last"]:.6f}')
    print(f'R_abs_err_mean : {result["summary"]["R_abs_err_mean"]:.6f}')
    print(f'figures/json: {out_prefix}_*.png / _summary.json')
    print('=' * 100)


if __name__ == '__main__':
    main()