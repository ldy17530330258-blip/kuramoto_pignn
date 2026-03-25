import os
import json
import pickle
import math
import random

import numpy as np
import torch
from torch_geometric.data import Data

from configs import kuramoto_config as config
from data_generation.generate_graphs import generate_graph_bank
from data_generation.simulate_kuramoto import simulate_kuramoto, graph_static_features


def set_seed(seed=config.GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_one_step_samples(graphs, metas):
    samples = []
    dt = float(config.DT)

    for gid, (G, meta) in enumerate(zip(graphs, metas)):
        N = G.number_of_nodes()
        omega = np.random.normal(0.0, config.OMEGA_STD, size=N).astype(np.float32)
        theta0 = np.random.uniform(-math.pi, math.pi, size=N).astype(np.float32)
        K = float(np.random.uniform(*config.K_RANGE))
        sim = simulate_kuramoto(G, K, omega, theta0)

        edges = np.array(list(G.edges()), dtype=np.int64)
        if edges.size == 0:
            continue

        # Undirected graph -> two directed edges
        edge_index = np.concatenate([edges.T, edges[:, ::-1].T], axis=1)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        deg, clust, alive_mask, omega_feat = graph_static_features(G, omega)

        for t_idx in range(len(sim['t']) - 1):
            theta_t = sim['theta'][t_idx]
            theta_next = sim['theta'][t_idx + 1]
            R_t = sim['R'][t_idx]
            Psi_t = sim['Psi'][t_idx]

            global_ctx = np.array([K, R_t, np.sin(Psi_t), np.cos(Psi_t)], dtype=np.float32)

            x = np.stack(
                [
                    omega_feat,
                    deg,
                    clust,
                    alive_mask,
                    np.sin(theta_t),
                    np.cos(theta_t),
                    np.full(N, K, dtype=np.float32),
                    np.full(N, R_t, dtype=np.float32),
                    np.full(N, np.sin(Psi_t), dtype=np.float32),
                    np.full(N, np.cos(Psi_t), dtype=np.float32),
                ],
                axis=1,
            )

            data = Data(
                x=torch.tensor(x, dtype=torch.float32),                         # [N, 10]
                edge_index=edge_index,                                          # [2, E]
                theta_t=torch.tensor(theta_t, dtype=torch.float32),             # [N]
                theta_next=torch.tensor(theta_next, dtype=torch.float32),       # [N]
                omega=torch.tensor(omega_feat, dtype=torch.float32),            # [N]
                alive_mask=torch.tensor(alive_mask, dtype=torch.float32),       # [N]
                K=torch.tensor([[K]], dtype=torch.float32),                     # [1, 1]
                dt=torch.tensor([[dt]], dtype=torch.float32),                   # [1, 1]
                global_ctx=torch.tensor(global_ctx, dtype=torch.float32).unsqueeze(0),  # [1, 4]
                R_next=torch.tensor([[sim['R'][t_idx + 1]]], dtype=torch.float32),      # [1, 1]
                graph_id=torch.tensor([gid], dtype=torch.long),                 # [1]
                time_id=torch.tensor([t_idx], dtype=torch.long),                # [1]
                net_type=meta['type'],
            )
            samples.append(data)

    return samples


def train_val_test_split(num_graphs: int):
    idx = np.arange(num_graphs)
    rng = np.random.RandomState(config.GLOBAL_SEED)
    rng.shuffle(idx)

    n_train = int(num_graphs * config.TRAIN_RATIO)
    n_val = int(num_graphs * config.VAL_RATIO)

    train_graphs = set(idx[:n_train])
    val_graphs = set(idx[n_train:n_train + n_val])
    test_graphs = set(idx[n_train + n_val:])
    return train_graphs, val_graphs, test_graphs


def save_dataset(samples, out_path=None):
    out_path = out_path or os.path.join(config.PYG_DIR, 'kuramoto_dataset.pt')
    torch.save(samples, out_path)
    return out_path


def build_and_save_dataset():
    set_seed()
    graphs, metas = generate_graph_bank()
    samples = build_one_step_samples(graphs, metas)
    path = save_dataset(samples)

    split = train_val_test_split(len(graphs))
    split_path = os.path.join(config.PYG_DIR, 'graph_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split, f)

    meta_path = os.path.join(config.PYG_DIR, 'graph_meta.json')
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(metas, f, indent=2)

    return path, split_path, meta_path
