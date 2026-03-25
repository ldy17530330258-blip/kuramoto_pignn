import math
import numpy as np
import networkx as nx
from scipy.integrate import solve_ivp

from configs import kuramoto_config as config



def kuramoto_ode(t, theta, omega, A, K):
    diff = theta[None, :] - theta[:, None]
    coupling = (A * np.sin(diff)).sum(axis=1)
    return omega + K * coupling



def simulate_kuramoto(G: nx.Graph, K: float, omega: np.ndarray, theta0: np.ndarray,
                      t_max=config.T_MAX, dt=config.DT):
    A = nx.to_numpy_array(G, dtype=float)
    t_eval = np.arange(0.0, t_max + 1e-12, dt)
    sol = solve_ivp(
        fun=lambda t, y: kuramoto_ode(t, y, omega, A, K),
        t_span=(0.0, t_max),
        y0=theta0,
        t_eval=t_eval,
        method='RK45',
        rtol=1e-6,
        atol=1e-8,
    )
    theta = sol.y.T.astype(np.float32)  # [T, N]
    complex_order = np.exp(1j * theta).mean(axis=1)
    R = np.abs(complex_order).astype(np.float32)
    Psi = np.angle(complex_order).astype(np.float32)
    return {'t': t_eval.astype(np.float32), 'theta': theta, 'R': R, 'Psi': Psi}



def graph_static_features(G: nx.Graph, omega: np.ndarray, alive_mask: np.ndarray | None = None):
    N = G.number_of_nodes()
    nodes = list(range(N))
    deg = np.array([G.degree(n) for n in nodes], dtype=np.float32)
    deg = deg / max(deg.max(), 1.0)
    clust_dict = nx.clustering(G)
    clust = np.array([clust_dict[n] for n in nodes], dtype=np.float32)
    if alive_mask is None:
        alive_mask = np.ones(N, dtype=np.float32)
    return deg, clust, alive_mask.astype(np.float32), omega.astype(np.float32)
