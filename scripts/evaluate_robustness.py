import os
import sys
import json
import math
import csv
import argparse
import random
import pickle
from collections import defaultdict

import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx
from scipy.integrate import solve_ivp
from torch_geometric.data import Data
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == 'scripts' else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import kuramoto_config as config
from models.kuramoto_model import KuramotoPIGNN
from models.kuramoto_model_v2 import KuramotoPIGNN_V2


# ============================================================
# basic utilities
# ============================================================
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


def safe_mean(vals):
    if len(vals) == 0:
        return float('nan')
    return float(np.mean(vals))


def safe_std(vals):
    if len(vals) == 0:
        return float('nan')
    return float(np.std(vals))


def fmt(x):
    if x != x:
        return 'nan'
    return f'{x:.6f}'


def canonical_attack_mode(mode: str) -> str:
    alias = {
        'random': 'random',
        'degree': 'highest_degree',
        'highest_degree': 'highest_degree',
        'betweenness': 'highest_betweenness',
        'highest_betweenness': 'highest_betweenness',
    }
    if mode not in alias:
        raise ValueError(f'Unsupported attack mode: {mode}')
    return alias[mode]


def parse_tags(args) -> list[str]:
    if args.tags is not None:
        tags = [x.strip() for x in args.tags.split(',') if x.strip()]
        tags = list(dict.fromkeys(tags))
        if len(tags) == 0:
            raise ValueError('--tags is empty after parsing')
        return tags
    return [args.tag]


def parse_q_values(args):
    if args.q_values is not None:
        vals = [float(x.strip()) for x in args.q_values.split(',') if x.strip() != '']
        vals = sorted(list(dict.fromkeys(vals)))
        return vals
    vals = np.arange(args.q_min, args.q_max + 1e-12, args.q_step).tolist()
    vals = [round(float(v), 10) for v in vals]
    return vals


# ============================================================
# angle / order parameter / residual utilities
# ============================================================
def compute_order_parameter_np(theta: np.ndarray):
    z = np.exp(1j * theta).mean()
    return float(np.abs(z)), float(np.angle(z))


def compute_order_parameter_masked_np(theta: np.ndarray, alive_mask: np.ndarray):
    active = alive_mask > 0.5
    if active.sum() == 0:
        return 0.0, 0.0
    return compute_order_parameter_np(theta[active])


def compute_order_parameter_masked_torch(theta: torch.Tensor, alive_mask: torch.Tensor):
    active = alive_mask > 0.5
    if int(active.sum().item()) == 0:
        return 0.0, 0.0
    z = torch.exp(1j * theta[active]).mean()
    return float(torch.abs(z).item()), float(torch.angle(z).item())


def circular_abs_error_masked(pred: np.ndarray, true: np.ndarray, alive_mask: np.ndarray) -> np.ndarray:
    active = alive_mask > 0.5
    if active.sum() == 0:
        return np.zeros(0, dtype=np.float32)
    diff = np.arctan2(np.sin(pred[active] - true[active]), np.cos(pred[active] - true[active]))
    return np.abs(diff).astype(np.float32)


def circular_mae_masked(pred: np.ndarray, true: np.ndarray, alive_mask: np.ndarray) -> float:
    err = circular_abs_error_masked(pred, true, alive_mask)
    if err.size == 0:
        return 0.0
    return float(err.mean())


def wrap_delta_np(theta_next: np.ndarray, theta_t: np.ndarray) -> np.ndarray:
    return np.arctan2(np.sin(theta_next - theta_t), np.cos(theta_next - theta_t)).astype(np.float32)


def attacked_rhs_from_theta(theta_t: np.ndarray, omega: np.ndarray, A_att: np.ndarray, K: float, alive_mask: np.ndarray):
    diff = theta_t[None, :] - theta_t[:, None]
    coupling = (A_att * np.sin(diff)).sum(axis=1)
    rhs = omega + K * coupling
    rhs = rhs * alive_mask
    return rhs.astype(np.float32)


def per_step_kuramoto_residual(theta_t: np.ndarray,
                               theta_next: np.ndarray,
                               omega: np.ndarray,
                               A_att: np.ndarray,
                               K: float,
                               dt: float,
                               alive_mask: np.ndarray) -> np.ndarray:
    delta = wrap_delta_np(theta_next, theta_t)
    lhs = delta / float(dt)
    rhs = attacked_rhs_from_theta(theta_t, omega, A_att, K, alive_mask)
    res = (lhs - rhs) * alive_mask
    return res.astype(np.float32)


def residual_series_stats(theta_series: np.ndarray,
                          omega: np.ndarray,
                          A_att: np.ndarray,
                          K: float,
                          dt: float,
                          alive_mask: np.ndarray):
    if theta_series.shape[0] < 2:
        empty = np.zeros((0, theta_series.shape[1]), dtype=np.float32)
        return {
            'residuals': empty,
            'abs_mean_per_step': np.zeros(0, dtype=np.float32),
            'rms_per_step': np.zeros(0, dtype=np.float32),
            'mean_abs': 0.0,
            'last_abs': 0.0,
            'mean_rms': 0.0,
            'last_rms': 0.0,
        }

    residuals = []
    active = alive_mask > 0.5
    for t in range(theta_series.shape[0] - 1):
        res = per_step_kuramoto_residual(
            theta_t=theta_series[t],
            theta_next=theta_series[t + 1],
            omega=omega,
            A_att=A_att,
            K=K,
            dt=dt,
            alive_mask=alive_mask,
        )
        residuals.append(res)

    residuals = np.stack(residuals, axis=0).astype(np.float32)
    if active.sum() == 0:
        abs_mean_per_step = np.zeros(residuals.shape[0], dtype=np.float32)
        rms_per_step = np.zeros(residuals.shape[0], dtype=np.float32)
    else:
        masked = residuals[:, active]
        abs_mean_per_step = np.mean(np.abs(masked), axis=1).astype(np.float32)
        rms_per_step = np.sqrt(np.mean(masked ** 2, axis=1)).astype(np.float32)

    return {
        'residuals': residuals,
        'abs_mean_per_step': abs_mean_per_step,
        'rms_per_step': rms_per_step,
        'mean_abs': float(abs_mean_per_step.mean()) if abs_mean_per_step.size > 0 else 0.0,
        'last_abs': float(abs_mean_per_step[-1]) if abs_mean_per_step.size > 0 else 0.0,
        'mean_rms': float(rms_per_step.mean()) if rms_per_step.size > 0 else 0.0,
        'last_rms': float(rms_per_step[-1]) if rms_per_step.size > 0 else 0.0,
    }


# ============================================================
# graph / attack utilities
# ============================================================
def directed_to_undirected_edges(edge_index: torch.Tensor):
    edges = edge_index.cpu().numpy().T
    undirected = set()
    for u, v in edges:
        a, b = (int(u), int(v)) if int(u) <= int(v) else (int(v), int(u))
        if a != b:
            undirected.add((a, b))
    return sorted(list(undirected))


def build_graph_from_undirected_edges(num_nodes: int, undirected_edges, alive_mask: np.ndarray | None = None):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    if alive_mask is None:
        G.add_edges_from(undirected_edges)
    else:
        active = alive_mask > 0.5
        G.add_edges_from([(u, v) for (u, v) in undirected_edges if active[u] and active[v]])
    return G


def directed_edge_index_from_graph(G: nx.Graph) -> torch.Tensor:
    edges = np.array(list(G.edges()), dtype=np.int64)
    if edges.size == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    edge_index = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
    return torch.tensor(edge_index, dtype=torch.long)


def attacked_static_features(G_att: nx.Graph, omega: np.ndarray, alive_mask: np.ndarray):
    N = G_att.number_of_nodes()
    nodes = list(range(N))
    deg = np.array([G_att.degree(n) for n in nodes], dtype=np.float32)
    deg = deg / max(float(deg.max()), 1.0)
    clust_dict = nx.clustering(G_att)
    clust = np.array([clust_dict[n] for n in nodes], dtype=np.float32)
    return deg, clust, alive_mask.astype(np.float32), omega.astype(np.float32)


def compute_attack_scores(G: nx.Graph):
    degree_scores = np.array([G.degree(n) for n in range(G.number_of_nodes())], dtype=np.float32)
    betweenness_dict = nx.betweenness_centrality(G, normalized=True)
    betweenness_scores = np.array([betweenness_dict[n] for n in range(G.number_of_nodes())], dtype=np.float32)
    return degree_scores, betweenness_scores


def sample_attack_mask(num_nodes: int,
                       q: float,
                       mode: str,
                       rng: np.random.RandomState,
                       degree_scores: np.ndarray | None = None,
                       betweenness_scores: np.ndarray | None = None):
    mode = canonical_attack_mode(mode)
    q = float(np.clip(q, 0.0, 1.0))
    k = int(round(q * num_nodes))
    alive = np.ones(num_nodes, dtype=np.float32)
    if k <= 0:
        return alive
    if k >= num_nodes:
        alive[:] = 0.0
        return alive

    if mode == 'random':
        attacked = rng.choice(num_nodes, size=k, replace=False)
    elif mode == 'highest_degree':
        if degree_scores is None:
            raise ValueError('degree_scores required for degree attack')
        attacked = np.argsort(-degree_scores)[:k]
    elif mode == 'highest_betweenness':
        if betweenness_scores is None:
            raise ValueError('betweenness_scores required for betweenness attack')
        attacked = np.argsort(-betweenness_scores)[:k]
    else:
        raise ValueError(f'Unsupported attack mode: {mode}')

    alive[attacked] = 0.0
    return alive


# ============================================================
# true attacked dynamics
# ============================================================
def attacked_kuramoto_rhs(theta: np.ndarray,
                          omega: np.ndarray,
                          A_base: np.ndarray,
                          K: float,
                          alive_mask: np.ndarray):
    A_att = A_base * alive_mask[:, None] * alive_mask[None, :]
    diff = theta[None, :] - theta[:, None]
    coupling = (A_att * np.sin(diff)).sum(axis=1)
    rhs = omega + K * coupling
    rhs = rhs * alive_mask
    return rhs


def simulate_attacked_kuramoto(A_base: np.ndarray,
                               omega: np.ndarray,
                               theta0: np.ndarray,
                               K: float,
                               alive_mask: np.ndarray,
                               rollout_steps: int,
                               dt: float):
    t_eval = np.arange(0.0, (rollout_steps + 1) * dt, dt, dtype=np.float32)
    sol = solve_ivp(
        fun=lambda t, y: attacked_kuramoto_rhs(y, omega, A_base, K, alive_mask),
        t_span=(0.0, float(rollout_steps) * dt),
        y0=theta0.astype(np.float64),
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8,
    )

    theta = sol.y.T.astype(np.float32)
    R = np.array([compute_order_parameter_masked_np(th, alive_mask)[0] for th in theta], dtype=np.float32)
    Psi = np.array([compute_order_parameter_masked_np(th, alive_mask)[1] for th in theta], dtype=np.float32)
    return {'t': t_eval, 'theta': theta, 'R': R, 'Psi': Psi}


# ============================================================
# model rollout under attack
# ============================================================
def make_attacked_model_input(theta_t: torch.Tensor,
                              edge_index: torch.Tensor,
                              omega: torch.Tensor,
                              deg: torch.Tensor,
                              clust: torch.Tensor,
                              alive_mask: torch.Tensor,
                              K: float,
                              dt: float,
                              device: torch.device) -> Data:
    R_scalar, psi_scalar = compute_order_parameter_masked_torch(theta_t, alive_mask)
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

    return Data(
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


def rollout_attacked_closed_loop(model,
                                 theta0: np.ndarray,
                                 omega: np.ndarray,
                                 K: float,
                                 dt: float,
                                 edge_index_att: torch.Tensor,
                                 deg_att: np.ndarray,
                                 clust_att: np.ndarray,
                                 alive_mask: np.ndarray,
                                 theta_true: np.ndarray,
                                 R_true: np.ndarray,
                                 A_att: np.ndarray,
                                 device: torch.device):
    rollout_steps = theta_true.shape[0] - 1

    theta_curr = torch.tensor(theta0, dtype=torch.float32, device=device)
    omega_t = torch.tensor(omega, dtype=torch.float32, device=device)
    alive_t = torch.tensor(alive_mask, dtype=torch.float32, device=device)
    deg_t = torch.tensor(deg_att, dtype=torch.float32, device=device)
    clust_t = torch.tensor(clust_att, dtype=torch.float32, device=device)
    edge_index_att = edge_index_att.to(device)

    theta_pred_list = [theta_curr.detach().cpu().numpy().astype(np.float32)]

    with torch.no_grad():
        for _ in range(rollout_steps):
            data = make_attacked_model_input(
                theta_t=theta_curr,
                edge_index=edge_index_att,
                omega=omega_t,
                deg=deg_t,
                clust=clust_t,
                alive_mask=alive_t,
                K=K,
                dt=dt,
                device=device,
            )
            out = model(data)
            theta_next = out['theta_pred_next']
            theta_curr = alive_t * theta_next + (1.0 - alive_t) * theta_curr
            theta_pred_list.append(theta_curr.detach().cpu().numpy().astype(np.float32))

    theta_pred = np.stack(theta_pred_list, axis=0)
    R_pred = np.array([compute_order_parameter_masked_np(th, alive_mask)[0] for th in theta_pred], dtype=np.float32)

    per_step_phase_mae = np.array([
        circular_mae_masked(theta_pred[t], theta_true[t], alive_mask) for t in range(theta_pred.shape[0])
    ], dtype=np.float32)
    per_step_R_abs_err = np.abs(R_pred - R_true).astype(np.float32)
    per_step_R_rel_err = (per_step_R_abs_err / np.maximum(np.abs(R_true), 1e-8)).astype(np.float32)

    pred_res = residual_series_stats(theta_pred, omega=omega, A_att=A_att, K=K, dt=dt, alive_mask=alive_mask)
    true_res = residual_series_stats(theta_true, omega=omega, A_att=A_att, K=K, dt=dt, alive_mask=alive_mask)

    return {
        'theta_pred': theta_pred,
        'R_pred': R_pred,
        'per_step_phase_mae': per_step_phase_mae,
        'per_step_R_abs_err': per_step_R_abs_err,
        'per_step_R_rel_err': per_step_R_rel_err,
        'pred_residual': pred_res,
        'true_residual': true_res,
    }


# ============================================================
# robustness evaluation core
# ============================================================
def extract_graph_bundle(samples):
    first = samples[0]
    N = int(first.theta_t.numel())
    theta0 = samples[0].theta_t.cpu().numpy().astype(np.float32)
    omega = first.omega.cpu().numpy().astype(np.float32)
    K = float(first.K.view(-1)[0].item())
    dt = float(first.dt.view(-1)[0].item())
    net_type = getattr(first, 'net_type', 'unknown')

    undirected_edges = directed_to_undirected_edges(first.edge_index)
    G = build_graph_from_undirected_edges(N, undirected_edges, alive_mask=None)
    A_base = nx.to_numpy_array(G, dtype=np.float32)
    degree_scores, betweenness_scores = compute_attack_scores(G)

    return {
        'num_nodes': N,
        'theta0': theta0,
        'omega': omega,
        'K': K,
        'dt': dt,
        'net_type': str(net_type),
        'graph': G,
        'A_base': A_base,
        'degree_scores': degree_scores,
        'betweenness_scores': betweenness_scores,
        'undirected_edges': undirected_edges,
    }


def prepare_attack_case(graph_bundle,
                        q: float,
                        rollout_steps: int,
                        attack_mode: str,
                        rng: np.random.RandomState):
    N = graph_bundle['num_nodes']
    theta0 = graph_bundle['theta0']
    omega = graph_bundle['omega']
    K = graph_bundle['K']
    dt = graph_bundle['dt']
    A_base = graph_bundle['A_base']

    alive_mask = sample_attack_mask(
        num_nodes=N,
        q=q,
        mode=attack_mode,
        rng=rng,
        degree_scores=graph_bundle['degree_scores'],
        betweenness_scores=graph_bundle['betweenness_scores'],
    )

    G_att = build_graph_from_undirected_edges(N, graph_bundle['undirected_edges'], alive_mask=alive_mask)
    edge_index_att = directed_edge_index_from_graph(G_att)
    deg_att, clust_att, _, _ = attacked_static_features(G_att, omega, alive_mask)
    A_att = A_base * alive_mask[:, None] * alive_mask[None, :]

    sim_true = simulate_attacked_kuramoto(
        A_base=A_base,
        omega=omega,
        theta0=theta0,
        K=K,
        alive_mask=alive_mask,
        rollout_steps=rollout_steps,
        dt=dt,
    )

    return {
        'q': float(q),
        'alive_mask': alive_mask,
        'A_att': A_att.astype(np.float32),
        'edge_index_att': edge_index_att,
        'deg_att': deg_att,
        'clust_att': clust_att,
        'sim_true': sim_true,
        'num_alive': int((alive_mask > 0.5).sum()),
    }


def evaluate_one_model_on_case(model,
                               tag: str,
                               graph_bundle,
                               attack_case,
                               tail_window: int,
                               device: torch.device):
    theta0 = graph_bundle['theta0']
    omega = graph_bundle['omega']
    K = graph_bundle['K']
    dt = graph_bundle['dt']
    net_type = graph_bundle['net_type']
    N = graph_bundle['num_nodes']

    alive_mask = attack_case['alive_mask']
    A_att = attack_case['A_att']
    sim_true = attack_case['sim_true']

    pred = rollout_attacked_closed_loop(
        model=model,
        theta0=theta0,
        omega=omega,
        K=K,
        dt=dt,
        edge_index_att=attack_case['edge_index_att'],
        deg_att=attack_case['deg_att'],
        clust_att=attack_case['clust_att'],
        alive_mask=alive_mask,
        theta_true=sim_true['theta'],
        R_true=sim_true['R'],
        A_att=A_att,
        device=device,
    )

    tail_window = min(int(tail_window), len(sim_true['R']))
    robust_true = float(np.mean(sim_true['R'][-tail_window:]))
    robust_pred = float(np.mean(pred['R_pred'][-tail_window:]))
    robust_abs_err = abs(robust_pred - robust_true)
    robust_rel_err = robust_abs_err / max(abs(robust_true), 1e-8)

    summary = {
        'model': tag,
        'q': float(attack_case['q']),
        'net_type': net_type,
        'num_nodes': int(N),
        'num_alive': int(attack_case['num_alive']),
        'phase_mae_mean': float(pred['per_step_phase_mae'].mean()),
        'phase_mae_last': float(pred['per_step_phase_mae'][-1]),
        'R_abs_err_mean': float(pred['per_step_R_abs_err'].mean()),
        'R_abs_err_last': float(pred['per_step_R_abs_err'][-1]),
        'R_rel_err_mean': float(pred['per_step_R_rel_err'].mean()),
        'R_rel_err_last': float(pred['per_step_R_rel_err'][-1]),
        'robust_true': robust_true,
        'robust_pred': robust_pred,
        'robust_abs_err': float(robust_abs_err),
        'robust_rel_err': float(robust_rel_err),
        'pred_phy_res_abs_mean': float(pred['pred_residual']['mean_abs']),
        'pred_phy_res_abs_last': float(pred['pred_residual']['last_abs']),
        'pred_phy_res_rms_mean': float(pred['pred_residual']['mean_rms']),
        'pred_phy_res_rms_last': float(pred['pred_residual']['last_rms']),
        'true_phy_res_abs_mean': float(pred['true_residual']['mean_abs']),
        'true_phy_res_abs_last': float(pred['true_residual']['last_abs']),
        'true_phy_res_rms_mean': float(pred['true_residual']['mean_rms']),
        'true_phy_res_rms_last': float(pred['true_residual']['last_rms']),
    }

    return {
        'summary': summary,
        'pred': pred,
    }


def aggregate_rows(detailed_rows, q_values, tags):
    out = []
    for tag in tags:
        model_rows = [r for r in detailed_rows if r['model'] == tag]
        for q in q_values:
            vals = [r for r in model_rows if abs(r['q'] - q) < 1e-12]
            out.append({
                'model': tag,
                'scope': 'overall',
                'net_type': 'ALL',
                'q': q,
                'count': len(vals),
                'robust_true_mean': safe_mean([v['robust_true'] for v in vals]),
                'robust_pred_mean': safe_mean([v['robust_pred'] for v in vals]),
                'robust_abs_err_mean': safe_mean([v['robust_abs_err'] for v in vals]),
                'robust_rel_err_mean': safe_mean([v['robust_rel_err'] for v in vals]),
                'phase_mae_mean': safe_mean([v['phase_mae_mean'] for v in vals]),
                'phase_mae_last': safe_mean([v['phase_mae_last'] for v in vals]),
                'R_abs_err_mean': safe_mean([v['R_abs_err_mean'] for v in vals]),
                'R_abs_err_last': safe_mean([v['R_abs_err_last'] for v in vals]),
                'R_rel_err_mean': safe_mean([v['R_rel_err_mean'] for v in vals]),
                'R_rel_err_last': safe_mean([v['R_rel_err_last'] for v in vals]),
                'pred_phy_res_abs_mean': safe_mean([v['pred_phy_res_abs_mean'] for v in vals]),
                'pred_phy_res_abs_last': safe_mean([v['pred_phy_res_abs_last'] for v in vals]),
                'pred_phy_res_rms_mean': safe_mean([v['pred_phy_res_rms_mean'] for v in vals]),
                'pred_phy_res_rms_last': safe_mean([v['pred_phy_res_rms_last'] for v in vals]),
                'true_phy_res_abs_mean': safe_mean([v['true_phy_res_abs_mean'] for v in vals]),
                'true_phy_res_abs_last': safe_mean([v['true_phy_res_abs_last'] for v in vals]),
                'true_phy_res_rms_mean': safe_mean([v['true_phy_res_rms_mean'] for v in vals]),
                'true_phy_res_rms_last': safe_mean([v['true_phy_res_rms_last'] for v in vals]),
            })
            for topo in ['ER', 'BA', 'WS']:
                vals_t = [r for r in vals if r['net_type'] == topo]
                out.append({
                    'model': tag,
                    'scope': 'topology',
                    'net_type': topo,
                    'q': q,
                    'count': len(vals_t),
                    'robust_true_mean': safe_mean([v['robust_true'] for v in vals_t]),
                    'robust_pred_mean': safe_mean([v['robust_pred'] for v in vals_t]),
                    'robust_abs_err_mean': safe_mean([v['robust_abs_err'] for v in vals_t]),
                    'robust_rel_err_mean': safe_mean([v['robust_rel_err'] for v in vals_t]),
                    'phase_mae_mean': safe_mean([v['phase_mae_mean'] for v in vals_t]),
                    'phase_mae_last': safe_mean([v['phase_mae_last'] for v in vals_t]),
                    'R_abs_err_mean': safe_mean([v['R_abs_err_mean'] for v in vals_t]),
                    'R_abs_err_last': safe_mean([v['R_abs_err_last'] for v in vals_t]),
                    'R_rel_err_mean': safe_mean([v['R_rel_err_mean'] for v in vals_t]),
                    'R_rel_err_last': safe_mean([v['R_rel_err_last'] for v in vals_t]),
                    'pred_phy_res_abs_mean': safe_mean([v['pred_phy_res_abs_mean'] for v in vals_t]),
                    'pred_phy_res_abs_last': safe_mean([v['pred_phy_res_abs_last'] for v in vals_t]),
                    'pred_phy_res_rms_mean': safe_mean([v['pred_phy_res_rms_mean'] for v in vals_t]),
                    'pred_phy_res_rms_last': safe_mean([v['pred_phy_res_rms_last'] for v in vals_t]),
                    'true_phy_res_abs_mean': safe_mean([v['true_phy_res_abs_mean'] for v in vals_t]),
                    'true_phy_res_abs_last': safe_mean([v['true_phy_res_abs_last'] for v in vals_t]),
                    'true_phy_res_rms_mean': safe_mean([v['true_phy_res_rms_mean'] for v in vals_t]),
                    'true_phy_res_rms_last': safe_mean([v['true_phy_res_rms_last'] for v in vals_t]),
                })
    return out


def print_summary_table(summary_rows, multi_model: bool):
    print('\n' + '=' * 180)
    print('ROBUSTNESS CURVE SUMMARY')
    print('=' * 180)
    if multi_model:
        header = (
            f"{'Model':<10} | {'Scope':<8} | {'Type':<6} | {'q':<6} | {'Count':<5} | {'TrueRob':<10} | {'PredRob':<10} | "
            f"{'AbsErr':<10} | {'RelErr':<10} | {'PhaseMean':<10} | {'RMean':<10} | {'PredRes':<10}"
        )
        print(header)
        print('-' * 180)
        for r in summary_rows:
            print(
                f"{r['model']:<10} | {r['scope']:<8} | {r['net_type']:<6} | {r['q']:<6.2f} | {r['count']:<5d} | "
                f"{fmt(r['robust_true_mean']):<10} | {fmt(r['robust_pred_mean']):<10} | {fmt(r['robust_abs_err_mean']):<10} | "
                f"{fmt(r['robust_rel_err_mean']):<10} | {fmt(r['phase_mae_mean']):<10} | {fmt(r['R_abs_err_mean']):<10} | "
                f"{fmt(r['pred_phy_res_abs_mean']):<10}"
            )
    else:
        header = (
            f"{'Scope':<8} | {'Type':<6} | {'q':<6} | {'Count':<5} | {'TrueRob':<10} | {'PredRob':<10} | "
            f"{'AbsErr':<10} | {'RelErr':<10} | {'PhaseMean':<10} | {'RMean':<10} | {'PredRes':<10}"
        )
        print(header)
        print('-' * 180)
        for r in summary_rows:
            print(
                f"{r['scope']:<8} | {r['net_type']:<6} | {r['q']:<6.2f} | {r['count']:<5d} | "
                f"{fmt(r['robust_true_mean']):<10} | {fmt(r['robust_pred_mean']):<10} | {fmt(r['robust_abs_err_mean']):<10} | "
                f"{fmt(r['robust_rel_err_mean']):<10} | {fmt(r['phase_mae_mean']):<10} | {fmt(r['R_abs_err_mean']):<10} | "
                f"{fmt(r['pred_phy_res_abs_mean']):<10}"
            )
    print('=' * 180)


# ============================================================
# output / plotting
# ============================================================
def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_curve_plots(summary_rows, out_prefix, q_values, tags):
    os.makedirs(os.path.dirname(out_prefix), exist_ok=True)

    overall_rows = [r for r in summary_rows if r['scope'] == 'overall']
    if len(overall_rows) == 0:
        return

    if len(tags) == 1:
        tag = tags[0]
        rows = sorted([r for r in overall_rows if r['model'] == tag], key=lambda x: x['q'])
        q = [r['q'] for r in rows]
        true_curve = [r['robust_true_mean'] for r in rows]
        pred_curve = [r['robust_pred_mean'] for r in rows]
        err_curve = [r['robust_abs_err_mean'] for r in rows]
        res_curve = [r['pred_phy_res_abs_mean'] for r in rows]

        plt.figure(figsize=(7, 4))
        plt.plot(q, true_curve, marker='o', label='ODE ground truth')
        plt.plot(q, pred_curve, marker='s', linestyle='--', label=f'{tag} prediction')
        plt.xlabel('attack ratio q')
        plt.ylabel('sync robustness score')
        plt.title('Overall synchronization robustness curve')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + '_overall_curve.png', dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(q, err_curve, marker='o')
        plt.xlabel('attack ratio q')
        plt.ylabel('|pred - true|')
        plt.title('Overall robustness curve absolute error')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_prefix + '_overall_error.png', dpi=180)
        plt.close()

        plt.figure(figsize=(7, 4))
        plt.plot(q, res_curve, marker='o')
        plt.xlabel('attack ratio q')
        plt.ylabel('physics residual mean')
        plt.title('Prediction trajectory physics residual')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(out_prefix + '_overall_residual.png', dpi=180)
        plt.close()

        for topo in ['ER', 'BA', 'WS']:
            rows_t = [r for r in summary_rows if r['scope'] == 'topology' and r['net_type'] == topo and r['model'] == tag]
            if len(rows_t) == 0:
                continue
            rows_t = sorted(rows_t, key=lambda x: x['q'])
            q = [r['q'] for r in rows_t]
            true_curve = [r['robust_true_mean'] for r in rows_t]
            pred_curve = [r['robust_pred_mean'] for r in rows_t]
            err_curve = [r['robust_abs_err_mean'] for r in rows_t]

            plt.figure(figsize=(7, 4))
            plt.plot(q, true_curve, marker='o', label='ODE ground truth')
            plt.plot(q, pred_curve, marker='s', linestyle='--', label=f'{tag} prediction')
            plt.xlabel('attack ratio q')
            plt.ylabel('sync robustness score')
            plt.title(f'{topo}: synchronization robustness curve')
            plt.grid(alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(out_prefix + f'_{topo}_curve.png', dpi=180)
            plt.close()

            plt.figure(figsize=(7, 4))
            plt.plot(q, err_curve, marker='o')
            plt.xlabel('attack ratio q')
            plt.ylabel('|pred - true|')
            plt.title(f'{topo}: robustness curve absolute error')
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.savefig(out_prefix + f'_{topo}_error.png', dpi=180)
            plt.close()
    else:
        baseline_rows = sorted([r for r in overall_rows if r['model'] == tags[0]], key=lambda x: x['q'])
        q = [r['q'] for r in baseline_rows]
        true_curve = [r['robust_true_mean'] for r in baseline_rows]

        plt.figure(figsize=(8, 5))
        plt.plot(q, true_curve, marker='o', linewidth=2.0, label='ODE ground truth')
        for tag in tags:
            rows_m = sorted([r for r in overall_rows if r['model'] == tag], key=lambda x: x['q'])
            pred_curve = [r['robust_pred_mean'] for r in rows_m]
            plt.plot(q, pred_curve, marker='s', linestyle='--', label=tag)
        plt.xlabel('attack ratio q')
        plt.ylabel('sync robustness score')
        plt.title('Overall synchronization robustness curve comparison')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + '_compare_overall_curve.png', dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for tag in tags:
            rows_m = sorted([r for r in overall_rows if r['model'] == tag], key=lambda x: x['q'])
            err_curve = [r['robust_abs_err_mean'] for r in rows_m]
            plt.plot(q, err_curve, marker='o', label=tag)
        plt.xlabel('attack ratio q')
        plt.ylabel('|pred - true|')
        plt.title('Overall robustness absolute error comparison')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + '_compare_overall_error.png', dpi=180)
        plt.close()

        plt.figure(figsize=(8, 5))
        for tag in tags:
            rows_m = sorted([r for r in overall_rows if r['model'] == tag], key=lambda x: x['q'])
            res_curve = [r['pred_phy_res_abs_mean'] for r in rows_m]
            plt.plot(q, res_curve, marker='o', label=tag)
        plt.xlabel('attack ratio q')
        plt.ylabel('physics residual mean')
        plt.title('Prediction trajectory physics residual comparison')
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_prefix + '_compare_overall_residual.png', dpi=180)
        plt.close()


# ============================================================
# main
# ============================================================
def main():
    parser = argparse.ArgumentParser(description='Evaluate synchronization robustness under node attacks.')
    parser.add_argument('--tag', type=str, default='v2_edge', help='Single model tag.')
    parser.add_argument('--tags', type=str, default=None,
                        help='Optional comma-separated tags for joint comparison, e.g. "pure_data,R_guided,v1,v2_edge"')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--graph_id', type=int, default=None,
                        help='Evaluate a single graph_id. If omitted, evaluate all graphs in the chosen split.')
    parser.add_argument('--attack_mode', type=str, default='random',
                        choices=['random', 'degree', 'highest_degree', 'betweenness', 'highest_betweenness'])
    parser.add_argument('--q_min', type=float, default=0.0)
    parser.add_argument('--q_max', type=float, default=0.5)
    parser.add_argument('--q_step', type=float, default=0.05)
    parser.add_argument('--q_values', type=str, default=None,
                        help='Optional explicit q list, e.g. "0,0.1,0.2,0.3,0.4,0.5"')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--rollout_steps', type=int, default=50)
    parser.add_argument('--tail_window', type=int, default=10,
                        help='Robustness score uses mean R(t) over last tail_window steps.')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out_prefix', type=str, default=None)
    args = parser.parse_args()

    set_seed(args.seed)
    device = pick_device(args.device)
    q_values = parse_q_values(args)
    tags = parse_tags(args)
    attack_mode = canonical_attack_mode(args.attack_mode)

    data_list, split_map = load_dataset_and_split()
    models = {}
    model_meta = {}
    for tag in tags:
        model, ckpt_path, ckpt = load_model(tag, device)
        model.eval()
        models[tag] = model
        model_meta[tag] = {
            'checkpoint': ckpt_path,
            'best_epoch': ckpt.get('epoch', None),
        }

    if args.graph_id is None:
        graph_ids = sorted(list(split_map[args.split]))
    else:
        if args.graph_id not in split_map[args.split]:
            raise ValueError(f'graph_id={args.graph_id} not in split={args.split}')
        graph_ids = [int(args.graph_id)]

    print('=' * 140)
    print('Starting synchronization robustness evaluation')
    print(f'models      : {tags}')
    print(f'device      : {device}')
    print(f'split       : {args.split}')
    print(f'graphs      : {len(graph_ids)}')
    print(f'attack_mode : {attack_mode}')
    print(f'q_values    : {q_values}')
    print(f'repeats     : {args.repeats}')
    print(f'rollout     : {args.rollout_steps}')
    print(f'tail_window : {args.tail_window}')
    for tag in tags:
        print(f'[{tag}] checkpoint: {model_meta[tag]["checkpoint"]} | best_epoch: {model_meta[tag]["best_epoch"]}')
    print('=' * 140)

    detailed_rows = []

    outer_pbar = tqdm(graph_ids, desc='Graphs', ascii=True)
    for gid in outer_pbar:
        samples = get_graph_samples(data_list, gid)
        graph_bundle = extract_graph_bundle(samples)

        for q in q_values:
            for rep in range(args.repeats):
                rng = np.random.RandomState(args.seed + gid * 1000 + int(round(q * 1000)) * 10 + rep)
                attack_case = prepare_attack_case(
                    graph_bundle=graph_bundle,
                    q=q,
                    rollout_steps=args.rollout_steps,
                    attack_mode=attack_mode,
                    rng=rng,
                )
                for tag in tags:
                    res = evaluate_one_model_on_case(
                        model=models[tag],
                        tag=tag,
                        graph_bundle=graph_bundle,
                        attack_case=attack_case,
                        tail_window=args.tail_window,
                        device=device,
                    )
                    s = res['summary']
                    detailed_rows.append({
                        'model': tag,
                        'graph_id': int(gid),
                        'repeat': int(rep),
                        'net_type': s['net_type'],
                        'q': float(q),
                        'num_nodes': int(s['num_nodes']),
                        'num_alive': int(s['num_alive']),
                        'robust_true': float(s['robust_true']),
                        'robust_pred': float(s['robust_pred']),
                        'robust_abs_err': float(s['robust_abs_err']),
                        'robust_rel_err': float(s['robust_rel_err']),
                        'phase_mae_mean': float(s['phase_mae_mean']),
                        'phase_mae_last': float(s['phase_mae_last']),
                        'R_abs_err_mean': float(s['R_abs_err_mean']),
                        'R_abs_err_last': float(s['R_abs_err_last']),
                        'R_rel_err_mean': float(s['R_rel_err_mean']),
                        'R_rel_err_last': float(s['R_rel_err_last']),
                        'pred_phy_res_abs_mean': float(s['pred_phy_res_abs_mean']),
                        'pred_phy_res_abs_last': float(s['pred_phy_res_abs_last']),
                        'pred_phy_res_rms_mean': float(s['pred_phy_res_rms_mean']),
                        'pred_phy_res_rms_last': float(s['pred_phy_res_rms_last']),
                        'true_phy_res_abs_mean': float(s['true_phy_res_abs_mean']),
                        'true_phy_res_abs_last': float(s['true_phy_res_abs_last']),
                        'true_phy_res_rms_mean': float(s['true_phy_res_rms_mean']),
                        'true_phy_res_rms_last': float(s['true_phy_res_rms_last']),
                        'scope': 'detailed',
                    })

    summary_rows = aggregate_rows(detailed_rows, q_values=q_values, tags=tags)
    print_summary_table(summary_rows, multi_model=(len(tags) > 1))

    out_prefix = args.out_prefix
    if out_prefix is None:
        if len(tags) == 1:
            out_prefix = os.path.join(config.LOG_DIR, f'robustness_{tags[0]}_{attack_mode}_{args.split}')
        else:
            tag_name = 'compare_' + '_'.join(tags)
            out_prefix = os.path.join(config.LOG_DIR, f'robustness_{tag_name}_{attack_mode}_{args.split}')

    fig_prefix = os.path.join(config.FIG_DIR, os.path.basename(out_prefix))
    save_curve_plots(summary_rows, out_prefix=fig_prefix, q_values=q_values, tags=tags)

    detailed_fieldnames = [
        'model', 'graph_id', 'repeat', 'net_type', 'q', 'num_nodes', 'num_alive',
        'robust_true', 'robust_pred', 'robust_abs_err', 'robust_rel_err',
        'phase_mae_mean', 'phase_mae_last',
        'R_abs_err_mean', 'R_abs_err_last', 'R_rel_err_mean', 'R_rel_err_last',
        'pred_phy_res_abs_mean', 'pred_phy_res_abs_last', 'pred_phy_res_rms_mean', 'pred_phy_res_rms_last',
        'true_phy_res_abs_mean', 'true_phy_res_abs_last', 'true_phy_res_rms_mean', 'true_phy_res_rms_last',
        'scope',
    ]
    save_csv(out_prefix + '_detailed.csv', detailed_rows, fieldnames=detailed_fieldnames)

    summary_fieldnames = [
        'model', 'scope', 'net_type', 'q', 'count',
        'robust_true_mean', 'robust_pred_mean', 'robust_abs_err_mean', 'robust_rel_err_mean',
        'phase_mae_mean', 'phase_mae_last',
        'R_abs_err_mean', 'R_abs_err_last', 'R_rel_err_mean', 'R_rel_err_last',
        'pred_phy_res_abs_mean', 'pred_phy_res_abs_last', 'pred_phy_res_rms_mean', 'pred_phy_res_rms_last',
        'true_phy_res_abs_mean', 'true_phy_res_abs_last', 'true_phy_res_rms_mean', 'true_phy_res_rms_last',
    ]
    save_csv(out_prefix + '_summary.csv', summary_rows, fieldnames=summary_fieldnames)

    summary_json = {
        'tags': tags,
        'split': args.split,
        'attack_mode': attack_mode,
        'q_values': q_values,
        'repeats': args.repeats,
        'rollout_steps': args.rollout_steps,
        'tail_window': args.tail_window,
        'models': model_meta,
        'summary_rows': summary_rows,
    }
    with open(out_prefix + '_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary_json, f, indent=2, ensure_ascii=False)

    print('\nSaved files:')
    print(f'  {out_prefix}_detailed.csv')
    print(f'  {out_prefix}_summary.csv')
    print(f'  {out_prefix}_summary.json')
    if len(tags) == 1:
        print(f'  {fig_prefix}_overall_curve.png')
        print(f'  {fig_prefix}_overall_error.png')
        print(f'  {fig_prefix}_overall_residual.png')
        for topo in ['ER', 'BA', 'WS']:
            print(f'  {fig_prefix}_{topo}_curve.png')
            print(f'  {fig_prefix}_{topo}_error.png')
    else:
        print(f'  {fig_prefix}_compare_overall_curve.png')
        print(f'  {fig_prefix}_compare_overall_error.png')
        print(f'  {fig_prefix}_compare_overall_residual.png')


if __name__ == '__main__':
    main()
