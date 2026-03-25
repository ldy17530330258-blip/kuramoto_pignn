import os
import sys
import json
import csv
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


DEFAULT_TAGS = ['pure_data', 'R_guided', 'v1', 'v2_edge']
DEFAULT_HORIZONS = [10, 20, 50]


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


def safe_std(vals):
    if len(vals) == 0:
        return float('nan')
    return float(np.std(vals))


def fmt(x):
    if x != x:  # nan
        return 'nan'
    return f'{x:.6f}'


def evaluate_one_model(tag, data_list, split_map, split_name, rollout_steps, device, horizons):
    model, ckpt_path, ckpt = load_model(tag, device)
    model.eval()

    graph_ids = sorted(list(split_map[split_name]))
    detailed_rows = []

    summary_by_type = defaultdict(lambda: {
        'phase_mae_mean': [],
        'phase_mae_last': [],
        'R_abs_err_mean': [],
        'R_abs_err_last': [],
    })

    horizon_by_type = defaultdict(lambda: defaultdict(list))

    pbar = tqdm(graph_ids, desc=f'[{tag}] graphs', ascii=True)

    skipped = 0
    for gid in pbar:
        samples = get_graph_samples(data_list, gid)
        actual_steps = min(int(rollout_steps), len(samples))

        if actual_steps < 1:
            skipped += 1
            continue

        res = rollout_closed_loop(model, samples, device, actual_steps)
        s = res['summary']
        net_type = s['net_type']
        if isinstance(net_type, list):
            net_type = net_type[0]

        summary_by_type[net_type]['phase_mae_mean'].append(s['phase_mae_mean'])
        summary_by_type[net_type]['phase_mae_last'].append(s['phase_mae_last'])
        summary_by_type[net_type]['R_abs_err_mean'].append(s['R_abs_err_mean'])
        summary_by_type[net_type]['R_abs_err_last'].append(s['R_abs_err_last'])

        for h in horizons:
            if h <= actual_steps:
                horizon_by_type[net_type][f'phase_h{h}'].append(float(res['per_step_phase_mae'][h]))
                horizon_by_type[net_type][f'R_err_h{h}'].append(float(res['per_step_R_abs_err'][h]))

        detailed_rows.append({
            'model': tag,
            'graph_id': int(gid),
            'split': split_name,
            'net_type': net_type,
            'num_nodes': int(s['num_nodes']),
            'rollout_steps': int(s['rollout_steps']),
            'phase_mae_mean': float(s['phase_mae_mean']),
            'phase_mae_last': float(s['phase_mae_last']),
            'R_abs_err_mean': float(s['R_abs_err_mean']),
            'R_abs_err_last': float(s['R_abs_err_last']),
        })

    overall = {
        'model': tag,
        'checkpoint': ckpt_path,
        'best_epoch': ckpt.get('epoch', None),
        'count': len(detailed_rows),
        'skipped': skipped,
        'phase_mae_mean_avg': safe_mean([r['phase_mae_mean'] for r in detailed_rows]),
        'phase_mae_mean_std': safe_std([r['phase_mae_mean'] for r in detailed_rows]),
        'phase_mae_last_avg': safe_mean([r['phase_mae_last'] for r in detailed_rows]),
        'phase_mae_last_std': safe_std([r['phase_mae_last'] for r in detailed_rows]),
        'R_abs_err_mean_avg': safe_mean([r['R_abs_err_mean'] for r in detailed_rows]),
        'R_abs_err_mean_std': safe_std([r['R_abs_err_mean'] for r in detailed_rows]),
        'R_abs_err_last_avg': safe_mean([r['R_abs_err_last'] for r in detailed_rows]),
        'R_abs_err_last_std': safe_std([r['R_abs_err_last'] for r in detailed_rows]),
    }

    topo_rows = []
    for net_type in sorted(summary_by_type.keys()):
        topo_rows.append({
            'model': tag,
            'net_type': net_type,
            'count': len(summary_by_type[net_type]['phase_mae_mean']),
            'phase_mae_mean_avg': safe_mean(summary_by_type[net_type]['phase_mae_mean']),
            'phase_mae_last_avg': safe_mean(summary_by_type[net_type]['phase_mae_last']),
            'R_abs_err_mean_avg': safe_mean(summary_by_type[net_type]['R_abs_err_mean']),
            'R_abs_err_last_avg': safe_mean(summary_by_type[net_type]['R_abs_err_last']),
        })

    horizon_rows = []
    for net_type in sorted(horizon_by_type.keys()):
        row = {
            'model': tag,
            'net_type': net_type,
        }
        for h in horizons:
            row[f'phase_h{h}'] = safe_mean(horizon_by_type[net_type][f'phase_h{h}'])
            row[f'R_err_h{h}'] = safe_mean(horizon_by_type[net_type][f'R_err_h{h}'])
        horizon_rows.append(row)

    return {
        'overall': overall,
        'topology_rows': topo_rows,
        'horizon_rows': horizon_rows,
        'detailed_rows': detailed_rows,
    }


def print_overall_table(overall_rows):
    print("\n" + "=" * 128)
    print("OVERALL MODEL COMPARISON")
    print("=" * 128)
    print(
        f"{'Model':<12} | {'Count':<5} | "
        f"{'PhaseMean':<12} | {'PhaseLast':<12} | "
        f"{'RMean':<12} | {'RLast':<12} | {'BestEp':<6}"
    )
    print("-" * 128)
    for r in overall_rows:
        print(
            f"{r['model']:<12} | {r['count']:<5d} | "
            f"{fmt(r['phase_mae_mean_avg']):<12} | {fmt(r['phase_mae_last_avg']):<12} | "
            f"{fmt(r['R_abs_err_mean_avg']):<12} | {fmt(r['R_abs_err_last_avg']):<12} | "
            f"{str(r['best_epoch']):<6}"
        )
    print("=" * 128)


def print_topology_table(topo_rows):
    print("\n" + "=" * 128)
    print("TOPOLOGY-WISE COMPARISON")
    print("=" * 128)
    print(
        f"{'Model':<12} | {'Type':<4} | {'Count':<5} | "
        f"{'PhaseMean':<12} | {'PhaseLast':<12} | "
        f"{'RMean':<12} | {'RLast':<12}"
    )
    print("-" * 128)
    for r in topo_rows:
        print(
            f"{r['model']:<12} | {r['net_type']:<4} | {r['count']:<5d} | "
            f"{fmt(r['phase_mae_mean_avg']):<12} | {fmt(r['phase_mae_last_avg']):<12} | "
            f"{fmt(r['R_abs_err_mean_avg']):<12} | {fmt(r['R_abs_err_last_avg']):<12}"
        )
    print("=" * 128)


def print_horizon_table(horizon_rows, horizons):
    print("\n" + "=" * 160)
    print("HORIZON-WISE COMPARISON")
    print("=" * 160)

    phase_cols = " | ".join([f"{('P@' + str(h)):<10}" for h in horizons])
    r_cols = " | ".join([f"{('R@' + str(h)):<10}" for h in horizons])
    print(f"{'Model':<12} | {'Type':<4} | {phase_cols} || {r_cols}")
    print("-" * 160)

    for r in horizon_rows:
        phase_vals = " | ".join([f"{fmt(r[f'phase_h{h}']):<10}" for h in horizons])
        r_vals = " | ".join([f"{fmt(r[f'R_err_h{h}']):<10}" for h in horizons])
        print(f"{r['model']:<12} | {r['net_type']:<4} | {phase_vals} || {r_vals}")

    print("=" * 160)


def save_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def save_markdown(path, overall_rows, topo_rows, horizon_rows, horizons, tags, split_name, rollout_steps):
    lines = []
    lines.append(f"# Rollout benchmark comparison")
    lines.append("")
    lines.append(f"- Split: `{split_name}`")
    lines.append(f"- Rollout steps: `{rollout_steps}`")
    lines.append(f"- Models: `{', '.join(tags)}`")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append("| Model | Count | PhaseMean | PhaseLast | RMean | RLast | BestEpoch |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for r in overall_rows:
        lines.append(
            f"| {r['model']} | {r['count']} | {fmt(r['phase_mae_mean_avg'])} | {fmt(r['phase_mae_last_avg'])} | "
            f"{fmt(r['R_abs_err_mean_avg'])} | {fmt(r['R_abs_err_last_avg'])} | {r['best_epoch']} |"
        )
    lines.append("")

    lines.append("## Topology-wise")
    lines.append("")
    lines.append("| Model | Type | Count | PhaseMean | PhaseLast | RMean | RLast |")
    lines.append("|---|---|---:|---:|---:|---:|---:|")
    for r in topo_rows:
        lines.append(
            f"| {r['model']} | {r['net_type']} | {r['count']} | {fmt(r['phase_mae_mean_avg'])} | "
            f"{fmt(r['phase_mae_last_avg'])} | {fmt(r['R_abs_err_mean_avg'])} | {fmt(r['R_abs_err_last_avg'])} |"
        )
    lines.append("")

    lines.append("## Horizon-wise")
    lines.append("")
    header = "| Model | Type | " + " | ".join([f"P@{h}" for h in horizons]) + " | " + " | ".join([f"R@{h}" for h in horizons]) + " |"
    sep = "|" + "---|" * (2 + len(horizons) * 2)
    lines.append(header)
    lines.append(sep)
    for r in horizon_rows:
        vals = [r['model'], r['net_type']]
        vals += [fmt(r[f'phase_h{h}']) for h in horizons]
        vals += [fmt(r[f'R_err_h{h}']) for h in horizons]
        lines.append("| " + " | ".join(vals) + " |")

    with open(path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))


def parse_args():
    parser = argparse.ArgumentParser(description='Run rollout benchmark for multiple checkpoints and generate comparison tables.')
    parser.add_argument('--tags', nargs='+', default=DEFAULT_TAGS,
                        help='model tags, e.g. pure_data R_guided v1 v2_edge')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--steps', type=int, default=50)
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--out_prefix', type=str, default='rollout_compare_all')
    return parser.parse_args()


def main():
    args = parse_args()
    device = pick_device(args.device)
    tags = args.tags
    split_name = args.split
    rollout_steps = int(args.steps)
    horizons = [h for h in DEFAULT_HORIZONS if h <= rollout_steps]

    print(f'Starting ALL-MODEL rollout comparison | device={device} | split={split_name} | steps={rollout_steps}')
    print(f'Model tags: {tags}')

    data_list, split_map = load_dataset_and_split()

    all_overall_rows = []
    all_topology_rows = []
    all_horizon_rows = []
    all_detailed_rows = []
    all_json = {
        'split': split_name,
        'rollout_steps': rollout_steps,
        'device': str(device),
        'tags': tags,
        'models': {},
    }

    for tag in tags:
        print(f"\n>>> Evaluating model: {tag}")
        res = evaluate_one_model(
            tag=tag,
            data_list=data_list,
            split_map=split_map,
            split_name=split_name,
            rollout_steps=rollout_steps,
            device=device,
            horizons=horizons,
        )

        all_overall_rows.append(res['overall'])
        all_topology_rows.extend(res['topology_rows'])
        all_horizon_rows.extend(res['horizon_rows'])
        all_detailed_rows.extend(res['detailed_rows'])

        all_json['models'][tag] = {
            'overall': res['overall'],
            'topology_rows': res['topology_rows'],
            'horizon_rows': res['horizon_rows'],
        }

    # print tables
    print_overall_table(all_overall_rows)
    print_topology_table(all_topology_rows)
    print_horizon_table(all_horizon_rows, horizons)

    # save files
    os.makedirs(config.LOG_DIR, exist_ok=True)
    prefix = os.path.join(config.LOG_DIR, args.out_prefix)

    save_csv(
        prefix + '_overall.csv',
        all_overall_rows,
        fieldnames=[
            'model', 'checkpoint', 'best_epoch', 'count', 'skipped',
            'phase_mae_mean_avg', 'phase_mae_mean_std',
            'phase_mae_last_avg', 'phase_mae_last_std',
            'R_abs_err_mean_avg', 'R_abs_err_mean_std',
            'R_abs_err_last_avg', 'R_abs_err_last_std',
        ],
    )

    save_csv(
        prefix + '_topology.csv',
        all_topology_rows,
        fieldnames=[
            'model', 'net_type', 'count',
            'phase_mae_mean_avg', 'phase_mae_last_avg',
            'R_abs_err_mean_avg', 'R_abs_err_last_avg',
        ],
    )

    horizon_fields = ['model', 'net_type']
    for h in horizons:
        horizon_fields.append(f'phase_h{h}')
    for h in horizons:
        horizon_fields.append(f'R_err_h{h}')

    save_csv(
        prefix + '_horizon.csv',
        all_horizon_rows,
        fieldnames=horizon_fields,
    )

    save_csv(
        prefix + '_detailed.csv',
        all_detailed_rows,
        fieldnames=[
            'model', 'graph_id', 'split', 'net_type', 'num_nodes',
            'rollout_steps', 'phase_mae_mean', 'phase_mae_last',
            'R_abs_err_mean', 'R_abs_err_last',
        ],
    )

    with open(prefix + '_summary.json', 'w', encoding='utf-8') as f:
        json.dump(all_json, f, indent=2, ensure_ascii=False)

    save_markdown(
        prefix + '_report.md',
        all_overall_rows,
        all_topology_rows,
        all_horizon_rows,
        horizons,
        tags,
        split_name,
        rollout_steps,
    )

    print(f"\nSaved files:")
    print(f"  {prefix}_overall.csv")
    print(f"  {prefix}_topology.csv")
    print(f"  {prefix}_horizon.csv")
    print(f"  {prefix}_detailed.csv")
    print(f"  {prefix}_summary.json")
    print(f"  {prefix}_report.md")


if __name__ == '__main__':
    main()