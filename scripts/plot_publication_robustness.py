import os
import re
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


MODEL_ORDER = ["pure_data", "R_guided", "v1", "v2_edge"]
MODEL_LABELS = {
    "pure_data": "Pure-data GNN",
    "R_guided": "Graph-level R-guided",
    "v1": "Node-residual v1",
    "v2_edge": "Edge-message v2",
}
MODEL_MARKERS = {
    "pure_data": "o",
    "R_guided": "s",
    "v1": "^",
    "v2_edge": "D",
}
TOPO_ORDER = ["ER", "BA", "WS"]
ATTACK_ALIAS = {
    "random": "Random attack",
    "degree": "Degree attack",
    "highest_degree": "Degree attack",
    "betweenness": "Betweenness attack",
    "highest_betweenness": "Betweenness attack",
}


def set_pub_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.2,
        "lines.markersize": 6.5,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.7,
        "savefig.dpi": 320,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.constrained_layout.use": True,
    })


def canonical_attack_name_from_path(path_str: str) -> str:
    name = Path(path_str).stem.lower()
    if "highest_degree" in name or "_degree_" in name or name.endswith("degree"):
        return "degree"
    if "highest_betweenness" in name or "betweenness" in name:
        return "betweenness"
    if "random" in name:
        return "random"
    return "unknown"


def load_one_result(summary_path: str, detailed_path: str | None = None, attack_name: str | None = None):
    summary_df = pd.read_csv(summary_path, encoding="utf-8-sig")
    detailed_df = None
    if detailed_path is not None and os.path.exists(detailed_path):
        detailed_df = pd.read_csv(detailed_path, encoding="utf-8-sig")

    if attack_name is None:
        attack_name = canonical_attack_name_from_path(summary_path)

    attack_title = ATTACK_ALIAS.get(attack_name, attack_name)

    return {
        "attack_name": attack_name,
        "attack_title": attack_title,
        "summary": summary_df,
        "detailed": detailed_df,
        "summary_path": summary_path,
        "detailed_path": detailed_path,
    }


def filter_rows(df: pd.DataFrame, scope: str, net_type: str | None = None):
    out = df[df["scope"] == scope].copy()
    if net_type is not None:
        out = out[out["net_type"] == net_type].copy()
    return out


def ordered_unique_q(df: pd.DataFrame):
    return sorted(df["q"].astype(float).unique().tolist())


def plot_overall_curve_panels(results: list[dict], save_dir: Path):
    n = len(results)
    fig, axes = plt.subplots(1, n, figsize=(6.2 * n, 4.6), squeeze=False)
    axes = axes[0]

    for ax, res in zip(axes, results):
        df = filter_rows(res["summary"], scope="overall", net_type="ALL")
        df = df.sort_values(["model", "q"])

        baseline = df[df["model"] == MODEL_ORDER[0]].sort_values("q")
        q = baseline["q"].to_numpy()
        true_curve = baseline["robust_true_mean"].to_numpy()

        ax.plot(q, true_curve, marker="o", linewidth=2.8, label="ODE ground truth")

        for model in MODEL_ORDER:
            rows = df[df["model"] == model].sort_values("q")
            if len(rows) == 0:
                continue
            ax.plot(
                rows["q"].to_numpy(),
                rows["robust_pred_mean"].to_numpy(),
                marker=MODEL_MARKERS[model],
                linestyle="--",
                label=MODEL_LABELS[model],
            )

        ax.set_title(res["attack_title"])
        ax.set_xlabel("Attack ratio q")
        ax.set_ylabel("Synchronization robustness")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle("Synchronization robustness curves under different attack protocols", y=1.12, fontsize=16)

    png_path = save_dir / "fig_overall_curve_panels.png"
    pdf_path = save_dir / "fig_overall_curve_panels.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def plot_topology_curve_1x3(result: dict, save_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6), squeeze=False)
    axes = axes[0]

    for ax, topo in zip(axes, TOPO_ORDER):
        df = filter_rows(result["summary"], scope="topology", net_type=topo)
        df = df.sort_values(["model", "q"])

        baseline = df[df["model"] == MODEL_ORDER[0]].sort_values("q")
        q = baseline["q"].to_numpy()
        true_curve = baseline["robust_true_mean"].to_numpy()

        ax.plot(q, true_curve, marker="o", linewidth=2.8, label="ODE ground truth")

        for model in MODEL_ORDER:
            rows = df[df["model"] == model].sort_values("q")
            if len(rows) == 0:
                continue
            ax.plot(
                rows["q"].to_numpy(),
                rows["robust_pred_mean"].to_numpy(),
                marker=MODEL_MARKERS[model],
                linestyle="--",
                label=MODEL_LABELS[model],
            )

        ax.set_title(f"{topo}")
        ax.set_xlabel("Attack ratio q")
        ax.set_ylabel("Synchronization robustness")
        ax.grid(True)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=True, bbox_to_anchor=(0.5, 1.05))
    fig.suptitle(f"Topology-wise robustness curves: {result['attack_title']}", y=1.12, fontsize=16)

    stem = f"fig_topology_curve_{result['attack_name']}"
    png_path = save_dir / f"{stem}.png"
    pdf_path = save_dir / f"{stem}.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def aggregate_from_detailed(detailed_df: pd.DataFrame):
    # 直接按 model 聚合，用于带误差棒柱状图
    grouped = (
        detailed_df.groupby("model", as_index=False)
        .agg(
            robust_abs_err_mean=("robust_abs_err", "mean"),
            robust_abs_err_std=("robust_abs_err", "std"),
            pred_res_mean=("pred_phy_res_abs_mean", "mean"),
            pred_res_std=("pred_phy_res_abs_mean", "std"),
            phase_mae_mean=("phase_mae_mean", "mean"),
            phase_mae_std=("phase_mae_mean", "std"),
        )
    )
    return grouped


def plot_bar_summary(results: list[dict], save_dir: Path):
    usable = [r for r in results if r["detailed"] is not None]
    if len(usable) == 0:
        print("[Warn] No detailed csv was provided. Skip bar-summary figure.")
        return

    attack_names = [r["attack_name"] for r in usable]
    attack_titles = [r["attack_title"] for r in usable]

    bar_width = 0.18
    x = np.arange(len(attack_names))

    fig, axes = plt.subplots(1, 2, figsize=(13.8, 4.8), squeeze=False)
    axes = axes[0]

    # 左图：robustness absolute error
    ax = axes[0]
    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for r in usable:
            agg = aggregate_from_detailed(r["detailed"])
            row = agg[agg["model"] == model]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row["robust_abs_err_mean"].iloc[0]))
                stds.append(float(row["robust_abs_err_std"].iloc[0]))
        ax.bar(x + (i - 1.5) * bar_width, means, width=bar_width, yerr=stds, capsize=3, label=MODEL_LABELS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(attack_titles)
    ax.set_ylabel("Mean robustness absolute error")
    ax.set_title("Robustness error comparison")
    ax.grid(True, axis="y")

    # 右图：physics residual
    ax = axes[1]
    for i, model in enumerate(MODEL_ORDER):
        means = []
        stds = []
        for r in usable:
            agg = aggregate_from_detailed(r["detailed"])
            row = agg[agg["model"] == model]
            if len(row) == 0:
                means.append(np.nan)
                stds.append(np.nan)
            else:
                means.append(float(row["pred_res_mean"].iloc[0]))
                stds.append(float(row["pred_res_std"].iloc[0]))
        ax.bar(x + (i - 1.5) * bar_width, means, width=bar_width, yerr=stds, capsize=3, label=MODEL_LABELS[model])

    ax.set_xticks(x)
    ax.set_xticklabels(attack_titles)
    ax.set_ylabel("Mean prediction physics residual")
    ax.set_title("Physics-consistency comparison")
    ax.grid(True, axis="y")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=True, bbox_to_anchor=(0.5, 1.06))
    fig.suptitle("Publication-style summary across attack protocols", y=1.12, fontsize=16)

    png_path = save_dir / "fig_bar_summary.png"
    pdf_path = save_dir / "fig_bar_summary.pdf"
    fig.savefig(png_path, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def export_compact_table(results: list[dict], save_dir: Path):
    rows = []
    for r in results:
        df = filter_rows(r["summary"], scope="overall", net_type="ALL")
        for model in MODEL_ORDER:
            sub = df[df["model"] == model]
            if len(sub) == 0:
                continue
            rows.append({
                "attack": r["attack_name"],
                "model": model,
                "mean_robust_abs_err_over_q": sub["robust_abs_err_mean"].mean(),
                "mean_robust_rel_err_over_q": sub["robust_rel_err_mean"].mean(),
                "mean_pred_residual_over_q": sub["pred_phy_res_abs_mean"].mean(),
                "mean_phase_mae_over_q": sub["phase_mae_mean"].mean(),
            })

    out_df = pd.DataFrame(rows)
    out_csv = save_dir / "paper_summary_table.csv"
    out_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[Saved] {out_csv}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate publication-style robustness figures from robustness csv files.")
    parser.add_argument(
        "--summary_paths",
        nargs="+",
        required=True,
        help="One or more *_summary.csv files. Example: random_summary.csv degree_summary.csv",
    )
    parser.add_argument(
        "--detailed_paths",
        nargs="*",
        default=None,
        help="Optional *_detailed.csv files, same order as summary_paths.",
    )
    parser.add_argument(
        "--attack_names",
        nargs="*",
        default=None,
        help="Optional explicit attack names, same order as summary_paths. Example: random degree",
    )
    parser.add_argument(
        "--topology_attack",
        type=str,
        default="degree",
        help="Which attack result to use for the 1x3 topology panel. Default: degree",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="outputs/figures/paper_ready",
        help="Directory to save publication-style figures.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    set_pub_style()

    summary_paths = args.summary_paths
    detailed_paths = args.detailed_paths or [None] * len(summary_paths)
    if len(detailed_paths) < len(summary_paths):
        detailed_paths = detailed_paths + [None] * (len(summary_paths) - len(detailed_paths))

    attack_names = args.attack_names or [None] * len(summary_paths)
    if len(attack_names) < len(summary_paths):
        attack_names = attack_names + [None] * (len(summary_paths) - len(attack_names))

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for sp, dp, an in zip(summary_paths, detailed_paths, attack_names):
        results.append(load_one_result(summary_path=sp, detailed_path=dp, attack_name=an))

    # 总体对照图：支持 1 个或多个 attack
    plot_overall_curve_panels(results, save_dir)

    # 拓扑 1x3 图：优先用指定 attack
    topo_result = None
    for r in results:
        if r["attack_name"] == args.topology_attack:
            topo_result = r
            break
    if topo_result is None:
        topo_result = results[0]
    plot_topology_curve_1x3(topo_result, save_dir)

    # 带误差棒的柱状图：依赖 detailed.csv
    plot_bar_summary(results, save_dir)

    # 导出一份紧凑汇总表
    export_compact_table(results, save_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()