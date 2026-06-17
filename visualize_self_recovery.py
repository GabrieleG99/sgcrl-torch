#!/usr/bin/env python3
"""Plot outputs from self_recovery_perturbation.py."""

from __future__ import annotations

import argparse
import csv
import math
import os
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


SUMMARY_NUMERIC_FIELDS = {
    "checkpoint_step",
    "episodes",
    "success_rate",
    "success_after_perturb_rate",
    "mean_recovery_steps",
    "mean_reset_dist",
    "mean_post_perturb_dist",
    "mean_final_dist",
    "mean_min_dist",
    "mean_post_perturb_min_dist",
    "mean_observed_displacement",
}
EPISODE_NUMERIC_FIELDS = {
    "checkpoint_step",
    "episode",
    "seed",
    "steps",
    "success",
    "success_before_perturb",
    "success_after_perturb",
    "first_success_step",
    "first_success_after_perturb_step",
    "recovery_steps",
    "reset_dist",
    "pre_perturb_dist",
    "post_perturb_dist",
    "final_dist",
    "min_dist",
    "post_perturb_min_dist",
    "perturb_step",
    "requested_dx",
    "requested_dy",
    "requested_dz",
    "observed_dx",
    "observed_dy",
    "observed_dz",
    "observed_displacement",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create publication-style summary plots from self_recovery_perturbation.py "
            "episodes.csv and summary.csv files."
        )
    )
    parser.add_argument("--input_dir", default="renders/self_recovery_perturbations")
    parser.add_argument("--summary_csv", default=None, help="Defaults to <input_dir>/summary.csv.")
    parser.add_argument("--episodes_csv", default=None, help="Defaults to <input_dir>/episodes.csv.")
    parser.add_argument("--output_dir", default=None, help="Defaults to <input_dir>/plots.")
    parser.add_argument("--labels", nargs="*", default=None, help="Optional labels to include, in plot order.")
    parser.add_argument("--settings", nargs="*", default=None, help="Optional settings to include, in plot order.")
    parser.add_argument("--format", choices=("png", "pdf", "svg"), default="png")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--title_prefix", default="")
    return parser.parse_args()


def maybe_float(value: str):
    if value is None or value == "":
        return np.nan
    try:
        return float(value)
    except ValueError:
        return value


def read_csv(path: Path, numeric_fields: set[str]) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="") as file:
        rows = []
        for row in csv.DictReader(file):
            converted = {}
            for key, value in row.items():
                converted[key] = maybe_float(value) if key in numeric_fields else value
            rows.append(converted)
    return rows


def finite_mean(values: Iterable[object]) -> float:
    numeric = [float(value) for value in values if is_finite_number(value)]
    return float(np.mean(numeric)) if numeric else np.nan


def is_finite_number(value: object) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number)


def finite_values(values: Iterable[object]) -> np.ndarray:
    return np.asarray([float(value) for value in values if is_finite_number(value)], dtype=float)


def build_summary_from_episodes(episode_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups = defaultdict(list)
    for row in episode_rows:
        groups[(row["label"], row["checkpoint"], row["env_name"], row["setting"])].append(row)

    summary = []
    for (label, checkpoint, env_name, setting), rows in sorted(groups.items()):
        summary.append(
            {
                "label": label,
                "checkpoint": checkpoint,
                "checkpoint_step": rows[0].get("checkpoint_step", np.nan),
                "env_name": env_name,
                "setting": setting,
                "episodes": len(rows),
                "success_rate": finite_mean(row.get("success") for row in rows),
                "success_after_perturb_rate": finite_mean(row.get("success_after_perturb") for row in rows),
                "mean_recovery_steps": finite_mean(row.get("recovery_steps") for row in rows),
                "mean_reset_dist": finite_mean(row.get("reset_dist") for row in rows),
                "mean_post_perturb_dist": finite_mean(row.get("post_perturb_dist") for row in rows),
                "mean_final_dist": finite_mean(row.get("final_dist") for row in rows),
                "mean_min_dist": finite_mean(row.get("min_dist") for row in rows),
                "mean_post_perturb_min_dist": finite_mean(row.get("post_perturb_min_dist") for row in rows),
                "mean_observed_displacement": finite_mean(row.get("observed_displacement") for row in rows),
            }
        )
    return summary


def unique_in_order(rows: Iterable[dict[str, object]], key: str) -> list[str]:
    seen = set()
    values = []
    for row in rows:
        value = str(row[key])
        if value not in seen:
            seen.add(value)
            values.append(value)
    return values


def filter_rows(
    rows: list[dict[str, object]],
    labels: list[str] | None,
    settings: list[str] | None,
) -> list[dict[str, object]]:
    out = rows
    if labels:
        label_set = set(labels)
        out = [row for row in out if row.get("label") in label_set]
    if settings:
        setting_set = set(settings)
        out = [row for row in out if row.get("setting") in setting_set]
    return out


def ordered_labels(rows: list[dict[str, object]], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    first_seen = {}
    min_steps = {}
    for row in rows:
        label = str(row["label"])
        first_seen.setdefault(label, len(first_seen))
        step = row.get("checkpoint_step", np.nan)
        if is_finite_number(step):
            min_steps[label] = min(min_steps.get(label, np.inf), float(step))
    if min_steps:
        return sorted(first_seen, key=lambda label: (min_steps.get(label, np.inf), first_seen[label]))
    return sorted(first_seen, key=lambda label: first_seen[label])


def ordered_settings(rows: list[dict[str, object]], requested: list[str] | None) -> list[str]:
    if requested:
        return requested
    preferred = ["none", "static", "dynamic"]
    present = set(unique_in_order(rows, "setting"))
    ordered = [setting for setting in preferred if setting in present]
    ordered.extend(setting for setting in unique_in_order(rows, "setting") if setting not in ordered)
    return ordered


def row_lookup(rows: list[dict[str, object]]) -> dict[tuple[str, str], dict[str, object]]:
    return {(str(row["label"]), str(row["setting"])): row for row in rows}


def grouped_bar_plot(
    rows: list[dict[str, object]],
    labels: list[str],
    settings: list[str],
    metric: str,
    ylabel: str,
    title: str,
    path: Path,
    *,
    ylim: tuple[float, float] | None = None,
    dpi: int = 180,
) -> None:
    data = row_lookup(rows)
    x = np.arange(len(labels), dtype=float)
    width = min(0.8 / max(len(settings), 1), 0.28)

    fig, ax = plt.subplots(figsize=(max(7, 1.2 * len(labels) + 2), 4.8), constrained_layout=True)
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(settings), 1)))
    for setting_idx, setting in enumerate(settings):
        offset = (setting_idx - (len(settings) - 1) / 2) * width
        values = []
        for label in labels:
            row = data.get((label, setting), {})
            value = row.get(metric, np.nan)
            values.append(float(value) if is_finite_number(value) else np.nan)
        bars = ax.bar(x + offset, values, width=width, label=setting, color=colors[setting_idx], edgecolor="0.25")
        for bar, value in zip(bars, values):
            if math.isfinite(value):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value,
                    f"{value:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(title="setting")
    if ylim is not None:
        ax.set_ylim(*ylim)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def distance_plot(
    rows: list[dict[str, object]],
    labels: list[str],
    settings: list[str],
    title: str,
    path: Path,
    *,
    dpi: int = 180,
) -> None:
    data = row_lookup(rows)
    metrics = [
        ("mean_reset_dist", "reset"),
        ("mean_post_perturb_dist", "after perturb"),
        ("mean_final_dist", "final"),
        ("mean_post_perturb_min_dist", "best after perturb"),
    ]
    groups = [(label, setting) for label in labels for setting in settings if (label, setting) in data]
    if not groups:
        return

    x = np.arange(len(groups), dtype=float)
    width = 0.18
    fig, ax = plt.subplots(figsize=(max(8, 0.8 * len(groups) + 3), 5), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(metrics)))
    for metric_idx, (metric, legend) in enumerate(metrics):
        offset = (metric_idx - (len(metrics) - 1) / 2) * width
        values = []
        for label, setting in groups:
            value = data[(label, setting)].get(metric, np.nan)
            values.append(float(value) if is_finite_number(value) else np.nan)
        ax.bar(x + offset, values, width=width, label=legend, color=colors[metric_idx], edgecolor="0.25")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{label}\n{setting}" for label, setting in groups], rotation=0)
    ax.set_ylabel("goal distance")
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def episode_distance_plot(
    episode_rows: list[dict[str, object]],
    labels: list[str],
    settings: list[str],
    title: str,
    path: Path,
    *,
    dpi: int = 180,
) -> None:
    rows = filter_rows(episode_rows, labels, settings)
    groups = [(label, setting) for label in labels for setting in settings]
    groups = [(label, setting) for label, setting in groups if any(row["label"] == label and row["setting"] == setting for row in rows)]
    if not groups:
        return

    fig, axes = plt.subplots(
        len(groups),
        1,
        figsize=(8, max(3.0 * len(groups), 4.0)),
        sharex=False,
        constrained_layout=True,
    )
    if len(groups) == 1:
        axes = [axes]

    for ax, (label, setting) in zip(axes, groups):
        group_rows = [row for row in rows if row["label"] == label and row["setting"] == setting]
        group_rows = sorted(group_rows, key=lambda row: float(row.get("episode", 0.0)))
        episodes = finite_values(row.get("episode") for row in group_rows)
        final_dist = finite_values(row.get("final_dist") for row in group_rows)
        post_min = finite_values(row.get("post_perturb_min_dist") for row in group_rows)
        success = [row for row in group_rows if float(row.get("success_after_perturb", 0.0)) >= 1.0]

        ax.plot(episodes[: len(final_dist)], final_dist, marker="o", linewidth=1.3, label="final")
        ax.plot(episodes[: len(post_min)], post_min, marker="o", linewidth=1.3, label="best after perturb")
        if success:
            success_episodes = finite_values(row.get("episode") for row in success)
            success_final = finite_values(row.get("final_dist") for row in success)
            ax.scatter(success_episodes[: len(success_final)], success_final, marker="*", s=80, color="black", label="success")
        ax.set_title(f"{label} / {setting}")
        ax.set_ylabel("goal distance")
        ax.grid(alpha=0.25)
        ax.legend(loc="best")
    axes[-1].set_xlabel("episode")
    fig.suptitle(title)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def perturbation_plot(
    episode_rows: list[dict[str, object]],
    labels: list[str],
    settings: list[str],
    title: str,
    path: Path,
    *,
    dpi: int = 180,
) -> None:
    rows = filter_rows(episode_rows, labels, [setting for setting in settings if setting != "none"])
    if not rows:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    groups = [(label, setting) for label in labels for setting in settings if setting != "none"]
    colors = plt.cm.Set2(np.linspace(0, 1, max(len(groups), 1)))

    for color, (label, setting) in zip(colors, groups):
        group_rows = [row for row in rows if row["label"] == label and row["setting"] == setting]
        if not group_rows:
            continue
        requested = np.sqrt(
            finite_values(row.get("requested_dx") for row in group_rows) ** 2
            + finite_values(row.get("requested_dy") for row in group_rows) ** 2
            + finite_values(row.get("requested_dz") for row in group_rows) ** 2
        )
        observed = finite_values(row.get("observed_displacement") for row in group_rows)
        final_dist = finite_values(row.get("final_dist") for row in group_rows)
        count = min(len(requested), len(observed), len(final_dist))
        if count == 0:
            continue
        axes[0].scatter(requested[:count], observed[:count], s=32, alpha=0.75, label=f"{label}/{setting}", color=color)
        axes[1].scatter(observed[:count], final_dist[:count], s=32, alpha=0.75, label=f"{label}/{setting}", color=color)

    axes[0].set_xlabel("requested displacement")
    axes[0].set_ylabel("observed displacement")
    axes[0].set_title("Perturbation magnitude")
    axes[0].grid(alpha=0.25)
    axes[1].set_xlabel("observed displacement")
    axes[1].set_ylabel("final goal distance")
    axes[1].set_title("Perturbation vs outcome")
    axes[1].grid(alpha=0.25)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    summary_csv = Path(args.summary_csv) if args.summary_csv else input_dir / "summary.csv"
    episodes_csv = Path(args.episodes_csv) if args.episodes_csv else input_dir / "episodes.csv"
    output_dir = Path(args.output_dir) if args.output_dir else input_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    episode_rows = read_csv(episodes_csv, EPISODE_NUMERIC_FIELDS)
    summary_rows = read_csv(summary_csv, SUMMARY_NUMERIC_FIELDS)
    if not summary_rows:
        if not episode_rows:
            raise FileNotFoundError(
                f"No summary rows found at {summary_csv} and no episode rows found at {episodes_csv}."
            )
        summary_rows = build_summary_from_episodes(episode_rows)

    labels = ordered_labels(summary_rows, args.labels)
    settings = ordered_settings(summary_rows, args.settings)
    summary_rows = filter_rows(summary_rows, labels, settings)
    episode_rows = filter_rows(episode_rows, labels, settings)
    if not summary_rows:
        raise ValueError("No rows remain after applying --labels/--settings filters.")

    prefix = f"{args.title_prefix} " if args.title_prefix else ""
    suffix = args.format
    grouped_bar_plot(
        summary_rows,
        labels,
        settings,
        "success_after_perturb_rate",
        "success rate",
        f"{prefix}Post-Perturbation Success",
        output_dir / f"success_after_perturb_rate.{suffix}",
        ylim=(0.0, 1.05),
        dpi=args.dpi,
    )
    grouped_bar_plot(
        summary_rows,
        labels,
        settings,
        "success_rate",
        "success rate",
        f"{prefix}Episode Success",
        output_dir / f"success_rate.{suffix}",
        ylim=(0.0, 1.05),
        dpi=args.dpi,
    )
    grouped_bar_plot(
        summary_rows,
        labels,
        settings,
        "mean_recovery_steps",
        "steps",
        f"{prefix}Recovery Latency",
        output_dir / f"mean_recovery_steps.{suffix}",
        dpi=args.dpi,
    )
    distance_plot(
        summary_rows,
        labels,
        settings,
        f"{prefix}Goal Distance Summary",
        output_dir / f"distance_summary.{suffix}",
        dpi=args.dpi,
    )

    if episode_rows:
        episode_distance_plot(
            episode_rows,
            labels,
            settings,
            f"{prefix}Per-Episode Distances",
            output_dir / f"episode_distances.{suffix}",
            dpi=args.dpi,
        )
        perturbation_plot(
            episode_rows,
            labels,
            settings,
            f"{prefix}Perturbation Diagnostics",
            output_dir / f"perturbation_diagnostics.{suffix}",
            dpi=args.dpi,
        )

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
