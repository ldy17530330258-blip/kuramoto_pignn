import os
import sys
import json
import argparse
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR) if os.path.basename(CURRENT_DIR) == 'scripts' else CURRENT_DIR
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from configs import kuramoto_config as config
from scripts.rollout_eval import (
    load_dataset_and_split,
    load_model,
    get_graph_samples,
    rollout_closed_loop,
)


def pick_device(device_arg: str) -> torch.device:
    if device_arg == 'cpu':
        return torch.device('cpu')
    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return torch.device('cuda')
        print('[Warn] CUDA unavailable, fallback to CPU.')
        return torch.device('cpu')
    return config.DEVICE


def safe_mean(vals):
    if len(vals) == 0:
        return float('nan')
    return float(np.mean(vals))


def print_summary_tables(results_by_type, horizon_results, horizons):
    print("\n" + "=" * 96)
    print("ROLLOUT BENCHMARK SUMMARY")
    print("=" * 96)
    print(f"{'Net Type':<10} | {'Count':<6} | {'Avg Phase MAE':<15} | {'Avg R Error':<15}")
    print("-" * 60)
    for net_type in sorted(results_by_type.keys()):
        metrics = results_by_type[net_type]
        count = len(metrics['phase_mae_mean'])
        p_mae = safe_mean(metrics['phase_mae_mean'])
        r_err = safe_mean(metrics['R_abs_err_mean'])
        print(f"{net_type:<10} | {count:<6} | {p_mae:<15.6f} | {r_err:<15.6f}")

    print("\n" + "-" * 96)
    print("ERROR AT SPECIFIC HORIZONS")
    print("-" * 96)

    left = " | ".join([f"Phase@{h:<2}".ljust(11) for h in horizons])
    right = " | ".join([f"Rerr@{h:<2}".ljust(11) for h in horizons])
    print(f"{'Net Type':<10} | {left} || {right}")
    print("-" * 96)

    for net_type in sorted(horizon_results.keys()):
        phase_parts = []
        r_parts = []
        for h in horizons:
            phase_parts.append(f"{safe_mean(horizon_results[net_type][f'phase_h{h}']):<11.6f}")
            r_parts.append(f"{safe_mean(horizon_results[net_type][f'R_err_h{h}']):<11.6f}")
        print(f"{net_type:<10} | " + " | ".join(phase_parts) + " || " + " | ".join(r_parts))

    print("=" * 96 + "\n")


def run_benchmark(tag='v2_edge', rollout_steps=50, device_arg='auto'):
    device = pick_device(device_arg)
    print(f"Starting batch rollout benchmark | model={tag} | device={device}")

    data_list, split_map = load_dataset_and_split()
    model, ckpt_path, ckpt = load_model(tag, device)
    model.eval()

    test_gids = sorted(list(split_map['test']))
    print(f"Total test graphs in split: {len(test_gids)}")
    print(f"Checkpoint: {ckpt_path}")
    print(f"Best epoch: {ckpt.get('epoch', 'NA')}")

    results_by_type = defaultdict(lambda: {
        'phase_mae_mean': [],
        'phase_mae_last': [],
        'R_abs_err_mean': [],
        'R_abs_err_last': [],
    })

    horizons = [10, 20, 50]
    horizon_results = defaultdict(lambda: defaultdict(list))

    skipped = 0

    for gid in tqdm(test_gids, desc="Evaluating Test Graphs", ascii=True):
        samples = get_graph_samples(data_list, gid)

        # 一张图如果只有很短的轨迹，就跳过超出可评估 horizon 的那部分
        actual_steps = min(rollout_steps, len(samples))
        if actual_steps < min(horizons):
            skipped += 1
            continue

        res = rollout_closed_loop(model, samples, device, actual_steps)
        net_type = res['summary']['net_type']
        if isinstance(net_type, list):
            net_type = net_type[0]

        results_by_type[net_type]['phase_mae_mean'].append(res['summary']['phase_mae_mean'])
        results_by_type[net_type]['phase_mae_last'].append(res['summary']['phase_mae_last'])
        results_by_type[net_type]['R_abs_err_mean'].append(res['summary']['R_abs_err_mean'])
        results_by_type[net_type]['R_abs_err_last'].append(res['summary']['R_abs_err_last'])

        for h in horizons:
            if h <= actual_steps:
                horizon_results[net_type][f'phase_h{h}'].append(float(res['per_step_phase_mae'][h]))
                horizon_results[net_type][f'R_err_h{h}'].append(float(res['per_step_R_abs_err'][h]))

    print(f"Skipped graphs (too short for min horizon): {skipped}")
    print_summary_tables(results_by_type, horizon_results, horizons)

    serializable = {
        'tag': tag,
        'rollout_steps': rollout_steps,
        'checkpoint': ckpt_path,
        'best_epoch': ckpt.get('epoch', None),
        'skipped': skipped,
        'summary_by_type': {},
        'horizon_by_type': {},
    }

    for net_type, metrics in results_by_type.items():
        serializable['summary_by_type'][net_type] = {
            'count': len(metrics['phase_mae_mean']),
            'phase_mae_mean_avg': safe_mean(metrics['phase_mae_mean']),
            'phase_mae_last_avg': safe_mean(metrics['phase_mae_last']),
            'R_abs_err_mean_avg': safe_mean(metrics['R_abs_err_mean']),
            'R_abs_err_last_avg': safe_mean(metrics['R_abs_err_last']),
        }

    for net_type, metrics in horizon_results.items():
        serializable['horizon_by_type'][net_type] = {}
        for h in horizons:
            serializable['horizon_by_type'][net_type][f'phase_h{h}'] = safe_mean(metrics[f'phase_h{h}'])
            serializable['horizon_by_type'][net_type][f'R_err_h{h}'] = safe_mean(metrics[f'R_err_h{h}'])

    os.makedirs(config.LOG_DIR, exist_ok=True)
    out_path = os.path.join(config.LOG_DIR, f'rollout_benchmark_{tag}.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(serializable, f, indent=2, ensure_ascii=False)

    print(f"Saved benchmark json to: {out_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tag', type=str, default='v2_edge')
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_benchmark(tag=args.tag, rollout_steps=args.steps, device_arg=args.device)