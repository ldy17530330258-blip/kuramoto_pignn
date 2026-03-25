import random
import numpy as np
import networkx as nx

from configs import kuramoto_config as config


def set_seed(seed=config.GLOBAL_SEED):
    random.seed(seed)
    np.random.seed(seed)



def _largest_cc_relabel(G: nx.Graph):
    if not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
    return nx.convert_node_labels_to_integers(G)



def generate_single_graph(net_type: str, rng: np.random.RandomState, seed: int):
    N = int(rng.randint(config.N_RANGE[0], config.N_RANGE[1] + 1))
    if net_type == 'ER':
        avg_deg = rng.uniform(*config.ER_AVG_DEGREE_RANGE)
        p = min(avg_deg / (N - 1), 1.0)
        G = nx.erdos_renyi_graph(N, p, seed=seed)
        meta = {'avg_degree_target': float(avg_deg), 'p': float(p)}
    elif net_type == 'BA':
        m = int(rng.randint(config.BA_M_RANGE[0], config.BA_M_RANGE[1] + 1))
        G = nx.barabasi_albert_graph(N, m, seed=seed)
        meta = {'m': int(m)}
    elif net_type == 'WS':
        k = int(rng.randint(config.WS_K_RANGE[0] // 2, config.WS_K_RANGE[1] // 2 + 1) * 2)
        p_rewire = float(rng.uniform(*config.WS_P_REWIRE_RANGE))
        G = nx.watts_strogatz_graph(N, k, p_rewire, seed=seed)
        meta = {'k': int(k), 'p_rewire': float(p_rewire)}
    else:
        raise ValueError(f'Unknown network type: {net_type}')
    G = _largest_cc_relabel(G)
    meta.update({'num_nodes': G.number_of_nodes(), 'num_edges': G.number_of_edges(), 'type': net_type})
    return G, meta



def generate_graph_bank():
    set_seed()
    rng = np.random.RandomState(config.GLOBAL_SEED)
    graphs, metas = [], []
    for t, net_type in enumerate(config.NETWORK_TYPES):
        for i in range(config.NUM_NETWORKS_PER_TYPE):
            G, meta = generate_single_graph(net_type, rng, seed=100000 * (t + 1) + i)
            graphs.append(G)
            metas.append(meta)
    return graphs, metas
