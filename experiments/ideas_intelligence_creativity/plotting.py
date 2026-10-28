"""
Visualization for Experiments 1 and 2.

Generates publication-quality plots for embedding in the blog post.
Uses dark background styling to match the PaperMod theme.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path

from .config import COLORS, PLOT_DEFAULTS
from .landscape import FitnessLandscape


def _apply_style():
    """Apply dark-background styling for blog integration."""
    plt.style.use(PLOT_DEFAULTS["style"])
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.facecolor": COLORS["background"],
        "axes.facecolor": "#2a2a2a",
        "savefig.facecolor": COLORS["background"],
    })


def plot_fitness_comparison(history_fixed: dict, history_meta: dict,
                            output_path: Path | None = None) -> plt.Figure:
    """
    Plot best fitness over time for fixed vs meta optimizer.

    This is the primary visualization: System A plateaus while System B
    shows discontinuous jumps.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=PLOT_DEFAULTS["figsize"], dpi=PLOT_DEFAULTS["dpi"])

    steps_f = range(len(history_fixed["best_fitness"]))
    steps_m = range(len(history_meta["best_fitness"]))

    ax.plot(steps_f, history_fixed["best_fitness"],
            color=COLORS["intelligence"], linewidth=2, alpha=0.9,
            label="Intelligence (fixed strategy)")
    ax.plot(steps_m, history_meta["best_fitness"],
            color=COLORS["creativity"], linewidth=2, alpha=0.9,
            label="Creativity (meta-strategy)")

    # Mark meta-events with vertical lines
    for event in history_meta.get("meta_events", []):
        ax.axvline(x=event["step"], color=COLORS["creativity"],
                   alpha=0.2, linestyle="--", linewidth=0.8)

    ax.set_xlabel("Optimization Step")
    ax.set_ylabel("Best Fitness")
    ax.set_title("Intelligence vs. Creativity: Fitness Over Time")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=PLOT_DEFAULTS["dpi"], bbox_inches="tight")
    return fig


def plot_landscape_trajectories(landscape: FitnessLandscape,
                                 history_fixed: dict, history_meta: dict,
                                 output_path: Path | None = None) -> plt.Figure:
    """
    Plot optimizer trajectories on the 2D fitness landscape contour.

    Shows how System A stays trapped in one basin while System B
    jumps between basins.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 8), dpi=PLOT_DEFAULTS["dpi"])

    X, Y, Z = landscape.get_grid(resolution=150)
    contour = ax.contourf(X, Y, Z, levels=30, cmap=COLORS["landscape"], alpha=0.8)
    plt.colorbar(contour, ax=ax, label="Fitness", shrink=0.8)

    # Fixed optimizer path
    pos_f = np.array(history_fixed["best_positions"])
    ax.plot(pos_f[:, 0], pos_f[:, 1], color=COLORS["intelligence"],
            linewidth=1.5, alpha=0.8, label="Intelligence")
    ax.scatter(pos_f[0, 0], pos_f[0, 1], color=COLORS["intelligence"],
               s=80, marker="o", zorder=5, edgecolors="white", linewidth=1)
    ax.scatter(pos_f[-1, 0], pos_f[-1, 1], color=COLORS["intelligence"],
               s=80, marker="*", zorder=5, edgecolors="white", linewidth=1)

    # Meta optimizer path
    pos_m = np.array(history_meta["best_positions"])
    ax.plot(pos_m[:, 0], pos_m[:, 1], color=COLORS["creativity"],
            linewidth=1.5, alpha=0.8, label="Creativity")
    ax.scatter(pos_m[0, 0], pos_m[0, 1], color=COLORS["creativity"],
               s=80, marker="o", zorder=5, edgecolors="white", linewidth=1)
    ax.scatter(pos_m[-1, 0], pos_m[-1, 1], color=COLORS["creativity"],
               s=80, marker="*", zorder=5, edgecolors="white", linewidth=1)

    # Mark global optimum
    opt_pos, _ = landscape.get_global_optimum()
    ax.scatter(opt_pos[0], opt_pos[1], color="gold", s=150, marker="X",
               zorder=10, edgecolors="white", linewidth=1.5, label="Global Optimum")

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Optimizer Trajectories on Fitness Landscape")
    ax.legend(loc="upper right")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=PLOT_DEFAULTS["dpi"], bbox_inches="tight")
    return fig


def plot_strategy_diversity(history_meta: dict,
                            output_path: Path | None = None) -> plt.Figure:
    """
    Plot strategy diversity (Shannon entropy) over time for the meta-optimizer.

    Strategy diversification is a leading indicator of fitness jumps.
    """
    _apply_style()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), dpi=PLOT_DEFAULTS["dpi"],
                                    sharex=True)

    steps = range(len(history_meta["best_fitness"]))

    # Top: fitness
    ax1.plot(steps, history_meta["best_fitness"],
             color=COLORS["creativity"], linewidth=2)
    ax1.set_ylabel("Best Fitness")
    ax1.set_title("Creativity: Fitness and Strategy Diversity")
    ax1.grid(True, alpha=0.2)

    # Bottom: strategy diversity
    ax2.plot(steps, history_meta["strategy_diversity"],
             color="#66CC99", linewidth=2, label="Strategy entropy")
    ax2.fill_between(steps, 0, history_meta["strategy_diversity"],
                     color="#66CC99", alpha=0.15)
    ax2.set_xlabel("Optimization Step")
    ax2.set_ylabel("Strategy Diversity (bits)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.2)

    # Mark meta-events
    for event in history_meta.get("meta_events", []):
        for ax in (ax1, ax2):
            ax.axvline(x=event["step"], color="white", alpha=0.15,
                       linestyle="--", linewidth=0.8)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=PLOT_DEFAULTS["dpi"], bbox_inches="tight")
    return fig


def plot_hierarchy_fitness(history_level0: dict, history_level1: dict,
                           history_level2: dict,
                           shift_points: list[int] | None = None,
                           output_path: Path | None = None) -> plt.Figure:
    """
    Plot fitness over time for all three hierarchy levels (Experiment 2).

    Level 0 is flat, Level 1 improves then degrades on shift,
    Level 2 recovers after each shift.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=PLOT_DEFAULTS["figsize"], dpi=PLOT_DEFAULTS["dpi"])

    ax.plot(history_level0["best_fitness"], color=COLORS["level_0"],
            linewidth=2, alpha=0.7, linestyle="--", label="Level 0 — Fixed (Idea)")
    ax.plot(history_level1["best_fitness"], color=COLORS["level_1"],
            linewidth=2, alpha=0.9, label="Level 1 — Intelligence")
    ax.plot(history_level2["best_fitness"], color=COLORS["level_2"],
            linewidth=2, alpha=0.9, label="Level 2 — Creativity")

    if shift_points:
        for sp in shift_points:
            ax.axvline(x=sp, color="white", alpha=0.3, linestyle=":",
                       linewidth=1.5)
            ax.text(sp + 2, ax.get_ylim()[1] * 0.95, "env shift",
                    color="white", alpha=0.5, fontsize=9)

    ax.set_xlabel("Generation")
    ax.set_ylabel("Best Fitness")
    ax.set_title("The Generator Hierarchy: Three Levels of Adaptation")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.2)

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=PLOT_DEFAULTS["dpi"], bbox_inches="tight")
    return fig


def plot_capability_timeline(events: list[dict],
                             output_path: Path | None = None) -> plt.Figure:
    """
    Plot a timeline of when new capabilities emerge in Level 2.

    events: list of dicts with keys "step", "capability", "description"
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(10, 4), dpi=PLOT_DEFAULTS["dpi"])

    if not events:
        ax.text(0.5, 0.5, "No capability events recorded",
                ha="center", va="center", color="white", fontsize=12)
        return fig

    steps = [e["step"] for e in events]
    labels = [e["capability"] for e in events]
    colors = plt.cm.Set2(np.linspace(0, 1, len(events)))

    ax.barh(range(len(events)), steps, color=colors, height=0.6, alpha=0.8)
    ax.set_yticks(range(len(events)))
    ax.set_yticklabels(labels)
    ax.set_xlabel("Generation")
    ax.set_title("Capability Emergence Timeline (Level 2)")
    ax.invert_yaxis()
    ax.grid(True, alpha=0.2, axis="x")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=PLOT_DEFAULTS["dpi"], bbox_inches="tight")
    return fig


def save_all_experiment1_plots(history_fixed: dict, history_meta: dict,
                                landscape: FitnessLandscape,
                                output_dir: Path) -> list[Path]:
    """Generate and save all Experiment 1 plots for the blog."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    p = output_dir / "fitness_comparison.png"
    plot_fitness_comparison(history_fixed, history_meta, p)
    paths.append(p)
    plt.close()

    p = output_dir / "landscape_trajectories.png"
    plot_landscape_trajectories(landscape, history_fixed, history_meta, p)
    paths.append(p)
    plt.close()

    p = output_dir / "strategy_diversity.png"
    plot_strategy_diversity(history_meta, p)
    paths.append(p)
    plt.close()

    return paths
