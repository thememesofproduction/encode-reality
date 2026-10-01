"""Rich visualization for epidemic modeling experiments."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from .config import COLORS, PLOT_DEFAULTS, SIRS_DEFAULTS, ABM_DEFAULTS


def _apply_style():
    sns.set_theme(style="darkgrid", palette="muted", rc={
        "figure.facecolor": "#1e1e1e",
        "axes.facecolor": "#2a2a2a",
        "axes.edgecolor": "#555555",
        "grid.color": "#3a3a3a",
        "text.color": "#e0e0e0",
        "axes.labelcolor": "#e0e0e0",
        "xtick.color": "#cccccc",
        "ytick.color": "#cccccc",
        "font.family": "sans-serif",
        "font.size": 12,
    })


def plot_sirs_recovery(ode_result, abm_ensemble, output_path=None,
                       n_steps=None, dt=None):
    """Plot ABM ensemble mean +/- std vs ODE solution with annotations.

    Args:
        ode_result: SIRSResult from sirs_ode.solve_sirs.
        abm_ensemble: list of SimulationHistory objects.
        output_path: Path to save figure (optional).
        n_steps: Number of ABM steps.
        dt: Time step size.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    if n_steps is None:
        n_steps = len(abm_ensemble[0].I_counts)
    if dt is None:
        dt = SIRS_DEFAULTS["t_max"] / n_steps

    fig, ax = plt.subplots(figsize=(12, 7))

    t_abm = np.arange(n_steps) * dt
    t_ode = ode_result.t
    N = ode_result.S[0] + ode_result.I[0] + ode_result.R[0]

    # Compute ensemble statistics (as percentage)
    all_S = np.array([h.S_counts for h in abm_ensemble]) / N * 100
    all_I = np.array([h.I_counts for h in abm_ensemble]) / N * 100
    all_R = np.array([h.R_counts for h in abm_ensemble]) / N * 100

    for data, color, label in [
        (all_S, COLORS["susceptible"], "S"),
        (all_I, COLORS["infected"], "I"),
        (all_R, COLORS["recovered"], "R"),
    ]:
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        ax.plot(t_abm, mean, color=color, linewidth=2, label=f"ABM {label} Mean")
        ax.fill_between(t_abm, mean - std, mean + std, color=color, alpha=0.2,
                        label=f"ABM {label} \u00b11\u03c3")

    # ODE reference (as percentage)
    ax.plot(t_ode, ode_result.S / N * 100, color=COLORS["ode"], linewidth=1.5,
            linestyle="--", alpha=0.8, label="ODE S")
    ax.plot(t_ode, ode_result.I / N * 100, color=COLORS["ode"], linewidth=1.5,
            linestyle="--", alpha=0.8, label="ODE I")
    ax.plot(t_ode, ode_result.R / N * 100, color=COLORS["ode"], linewidth=1.5,
            linestyle="--", alpha=0.8, label="ODE R")

    # Annotations
    mean_I = all_I.mean(axis=0)
    peak_idx = np.argmax(mean_I)
    peak_val = mean_I[peak_idx]
    peak_day = t_abm[peak_idx]

    ax.annotate(
        f"Peak infection: ~{peak_val:.0f}% of population",
        xy=(peak_day, peak_val),
        xytext=(peak_day + 30, peak_val + 10),
        fontsize=10, color="#e0e0e0",
        arrowprops=dict(arrowstyle="->", color="#cccccc", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333", edgecolor="#555555", alpha=0.9),
    )

    n_runs = len(abm_ensemble)
    info_text = f"{n_runs} independent ABM simulations\nHomogeneous mixing, N={int(N):,}"
    ax.text(0.02, 0.97, info_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", color="#cccccc",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333",
                      edgecolor="#555555", alpha=0.9))

    ax.set_xlabel("Time (days)", fontsize=13)
    ax.set_ylabel("Population (%)", fontsize=13)
    ax.set_title("SIRS Recovery: ABM Ensemble vs. ODE Solution", fontsize=15, pad=10)
    ax.set_ylim(0, 100)

    # Subtitle
    fig.text(0.5, 0.01,
             "Under identical assumptions, the agent-based model recovers the smooth ODE prediction",
             ha="center", fontsize=10, color="#999999", style="italic")

    # Custom legend
    handles = [
        plt.Line2D([0], [0], color=COLORS["susceptible"], lw=2, label="ABM S"),
        plt.Line2D([0], [0], color=COLORS["infected"], lw=2, label="ABM I"),
        plt.Line2D([0], [0], color=COLORS["recovered"], lw=2, label="ABM R"),
        mpatches.Patch(color=COLORS["infected"], alpha=0.2, label="ABM \u00b11\u03c3"),
        plt.Line2D([0], [0], color=COLORS["ode"], lw=1.5, linestyle="--", label="ODE Prediction"),
    ]
    ax.legend(handles=handles, loc="center right", fontsize=9,
              facecolor="#2a2a2a", edgecolor="#555555")

    plt.subplots_adjust(bottom=0.12)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_network_divergence(ode_result, recovery_ensemble, network_runs,
                            output_path=None, n_steps=None, dt=None):
    """Plot 2-panel: recovery (left) vs network divergence (right).

    Args:
        ode_result: SIRSResult from sirs_ode.solve_sirs.
        recovery_ensemble: list of SimulationHistory (homogeneous mixing).
        network_runs: list of SimulationHistory from network ABM.
        output_path: Path to save figure.
        n_steps: Number of ABM steps.
        dt: Time step size.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    if n_steps is None:
        n_steps = len(network_runs[0].I_counts)
    if dt is None:
        dt = SIRS_DEFAULTS["t_max"] / n_steps

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(32, 18), sharey=True)

    t_abm = np.arange(n_steps) * dt
    t_ode = ode_result.t
    N = ode_result.S[0] + ode_result.I[0] + ode_result.R[0]

    # --- Left panel: Recovery (infected only) ---
    all_I_recovery = np.array([h.I_counts for h in recovery_ensemble]) / N * 100
    mean_I = all_I_recovery.mean(axis=0)
    std_I = all_I_recovery.std(axis=0)

    ax_left.plot(t_abm, mean_I, color=COLORS["infected"], linewidth=4, label="ABM Mean")
    ax_left.fill_between(t_abm, mean_I - std_I, mean_I + std_I,
                         color=COLORS["infected"], alpha=0.2, label="ABM \u00b11\u03c3")
    ax_left.plot(t_ode, ode_result.I / N * 100, color=COLORS["ode"], linewidth=4,
                 linestyle="--", label="ODE Prediction")

    ax_left.set_xlabel("Time (days)", fontsize=37)
    ax_left.set_ylabel("Infected (%)", fontsize=37)
    ax_left.set_title("Homogeneous Mixing (Recovery)", fontsize=40)
    ax_left.legend(fontsize=28, facecolor="#2a2a2a", edgecolor="#555555")
    ax_left.tick_params(labelsize=28)

    # --- Right panel: Network divergence ---
    # Individual trajectories with varied colors
    n_runs = len(network_runs)
    cmap = plt.cm.cool
    for i, hist in enumerate(network_runs):
        color = cmap(i / max(n_runs - 1, 1))
        label = "Individual runs" if i == 0 else None
        ax_right.plot(t_abm, hist.I_counts / N * 100, color=color,
                      alpha=0.4, linewidth=1.8, label=label)

    # ABM ensemble mean
    all_I_net = np.array([h.I_counts for h in network_runs]) / N * 100
    mean_I_net = all_I_net.mean(axis=0)
    ax_right.plot(t_abm, mean_I_net, color=COLORS["network_abm"],
                  linewidth=5, label="ABM Mean")

    # ODE reference
    ax_right.plot(t_ode, ode_result.I / N * 100, color=COLORS["ode"], linewidth=4,
                  linestyle="--", label="ODE Prediction")

    # Annotations for right panel
    # Show spread of trajectories
    std_I_net = all_I_net.std(axis=0)
    max_spread_idx = np.argmax(std_I_net)
    spread_day = t_abm[max_spread_idx]
    spread_top = mean_I_net[max_spread_idx] + std_I_net[max_spread_idx]

    ax_right.annotate(
        "Stochastic variability \u2014\neach network realization\nproduces different dynamics",
        xy=(spread_day, spread_top),
        xytext=(spread_day + 40, spread_top + 5),
        fontsize=28, color="#e0e0e0",
        arrowprops=dict(arrowstyle="->", color="#cccccc", lw=3),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#333333",
                  edgecolor="#555555", alpha=0.9),
    )

    # ODE annotation
    ode_peak_idx = np.argmax(ode_result.I)
    ode_peak_day = t_ode[ode_peak_idx]
    ode_peak_val = ode_result.I[ode_peak_idx] / N * 100
    ax_right.annotate(
        "ODE prediction\n(assumes homogeneous mixing)",
        xy=(ode_peak_day, ode_peak_val),
        xytext=(ode_peak_day + 50, ode_peak_val - 8),
        fontsize=28, color="#e0e0e0",
        arrowprops=dict(arrowstyle="->", color="#cccccc", lw=3),
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#333333",
                  edgecolor="#555555", alpha=0.9),
    )

    # Info text
    info = "Same rates: \u03b2=0.3, \u03b3=0.1, \u03c9=0.01\nBarab\u00e1si-Albert network (m=3)\nHeterogeneous transmission"
    ax_right.text(0.98, 0.97, info,
                  transform=ax_right.transAxes, fontsize=24,
                  verticalalignment="top", horizontalalignment="right", color="#cccccc",
                  bbox=dict(boxstyle="round,pad=0.5", facecolor="#333333",
                            edgecolor="#555555", alpha=0.9))

    ax_right.set_xlabel("Time (days)", fontsize=37)
    ax_right.set_title("Scale-Free Network (Divergence)", fontsize=40)
    ax_right.legend(fontsize=28, facecolor="#2a2a2a", edgecolor="#555555", loc="center right")
    ax_right.tick_params(labelsize=28)

    # Shared subtitle
    fig.suptitle("Network Structure Breaks ODE Equivalence", fontsize=46, y=1.02,
                 color="#e0e0e0")
    fig.text(0.5, -0.02,
             "Same SIRS rates, same population \u2014 network structure alone breaks the equivalence",
             ha="center", fontsize=32, color="#999999", style="italic")

    plt.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_irreducibility_sweep(sweep_values, sweep_results, output_path=None,
                              n_ensemble=None):
    """Plot parameter sweep showing nonlinear growth in outbreak size vs non-compliance.

    Args:
        sweep_values: list of non-compliance rate values (0 = full compliance, 1 = none).
        sweep_results: dict with 'peak_mean', 'peak_std' arrays.
        output_path: Path to save figure.

    Returns:
        matplotlib Figure.
    """
    _apply_style()
    fig, ax = plt.subplots(figsize=(12, 7))

    # Convert compliance → non-compliance and reverse so x runs 0 → 1
    means = np.array(sweep_results["peak_mean"])[::-1] * 100
    stds = np.array(sweep_results["peak_std"])[::-1] * 100
    x = 1.0 - np.array(sweep_values)[::-1]

    ax.plot(x, means, color=COLORS["sweep_line"], linewidth=2.5,
            marker="o", markersize=8, label="Peak infection (mean)", zorder=5)
    ax.fill_between(x, means - stds, means + stds,
                    color=COLORS["sweep_line"], alpha=0.2, label="\u00b11\u03c3")

    # Region annotations
    ax.text(0.15, 0.35, "Low non-compliance \u2192\nEpidemic contained",
            transform=ax.transAxes, fontsize=10, color="#66CC99",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333",
                      edgecolor="#66CC99", alpha=0.9))

    ax.text(0.85, 0.85, "High non-compliance \u2192\nLarge outbreaks",
            transform=ax.transAxes, fontsize=10, color="#CC6666",
            ha="center", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333",
                      edgecolor="#CC6666", alpha=0.9))

    # Info text
    ax.text(0.02, 0.97,
            f"{n_ensemble or '?'} simulations per non-compliance level\nBarab\u00e1si-Albert network, N=1,000",
            transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="left", color="#cccccc",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333",
                      edgecolor="#555555", alpha=0.9))

    ax.set_xlabel("Non-Compliance Rate", fontsize=13)
    ax.set_ylabel("Peak Infection (% of population)", fontsize=13)
    ax.set_title("Outbreak Size vs. Non-Compliance: Parameter Sweep", fontsize=15, pad=10)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(bottom=0)

    # Subtitle
    fig.text(0.5, 0.01,
             "Peak outbreak size grows nonlinearly with non-compliance rate \u2014 "
             "a signature of computational irreducibility",
             ha="center", fontsize=10, color="#999999", style="italic",
             wrap=True)

    ax.legend(fontsize=10, facecolor="#2a2a2a", edgecolor="#555555", loc="upper center")
    plt.subplots_adjust(bottom=0.12)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def save_all_plots(ode_result, recovery_ensemble, network_runs,
                   sweep_values, sweep_results, output_dir,
                   n_steps=None, dt=None, n_ensemble=None):
    """Generate and save all experiment plots.

    Returns:
        List of saved file Paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig = plot_sirs_recovery(ode_result, recovery_ensemble,
                             output_dir / "sirs_recovery.png",
                             n_steps=n_steps, dt=dt)
    plt.close(fig)
    saved.append(output_dir / "sirs_recovery.png")

    fig = plot_network_divergence(ode_result, recovery_ensemble, network_runs,
                                 output_dir / "network_divergence.png",
                                 n_steps=n_steps, dt=dt)
    plt.close(fig)
    saved.append(output_dir / "network_divergence.png")

    fig = plot_irreducibility_sweep(sweep_values, sweep_results,
                                   output_dir / "irreducibility_sweep.png",
                                   n_ensemble=n_ensemble)
    plt.close(fig)
    saved.append(output_dir / "irreducibility_sweep.png")

    return saved


def plot_structured_comparison(structured_data, output_path=None,
                                n_steps=None, dt=None):
    """Plot 3-curve trajectory comparison: ODE vs homogeneous ABM vs layered ABM.

    Args:
        structured_data: dict with 'ode', 'homogeneous_ensemble', 'layered_ensemble'.
        output_path: Path to save figure.
        n_steps: Number of ABM steps.
        dt: Time step size.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    ode = structured_data["ode"]
    homo = structured_data["homogeneous_ensemble"]
    layered = structured_data["layered_ensemble"]

    if n_steps is None:
        n_steps = len(homo[0].I_counts)
    if dt is None:
        dt = SIRS_DEFAULTS["t_max"] / n_steps

    fig, ax = plt.subplots(figsize=(14, 8))

    t_abm = np.arange(n_steps) * dt
    t_ode = ode.t
    N = ode.S[0] + ode.I[0] + ode.R[0]

    # ODE (dashed white)
    ax.plot(t_ode, ode.I / N * 100, color=COLORS["ode"], linewidth=2.5,
            linestyle="--", label="ODE (deterministic)", zorder=5)

    # Homogeneous ABM mean + band
    homo_I = np.array([h.I_counts for h in homo]) / N * 100
    homo_mean = homo_I.mean(axis=0)
    homo_std = homo_I.std(axis=0)
    ax.plot(t_abm, homo_mean, color=COLORS["infected"], linewidth=2.5,
            label="Homogeneous ABM", zorder=4)
    ax.fill_between(t_abm, homo_mean - homo_std, homo_mean + homo_std,
                    color=COLORS["infected"], alpha=0.15)

    # Layered ABM mean + band
    layered_I = np.array([h.I_counts for h in layered]) / N * 100
    layered_mean = layered_I.mean(axis=0)
    layered_std = layered_I.std(axis=0)
    ax.plot(t_abm, layered_mean, color="#E8A838", linewidth=2.5,
            label="Layered Network ABM", zorder=4)
    ax.fill_between(t_abm, layered_mean - layered_std, layered_mean + layered_std,
                    color="#E8A838", alpha=0.15)

    # Peak annotations
    homo_peak_idx = np.argmax(homo_mean)
    homo_peak_val = homo_mean[homo_peak_idx]
    homo_peak_day = t_abm[homo_peak_idx]

    layered_peak_idx = np.argmax(layered_mean)
    layered_peak_val = layered_mean[layered_peak_idx]
    layered_peak_day = t_abm[layered_peak_idx]

    ax.annotate(
        f"Homogeneous peak: {homo_peak_val:.0f}%\n(day {homo_peak_day:.0f})",
        xy=(homo_peak_day, homo_peak_val),
        xytext=(homo_peak_day + 40, homo_peak_val - 8),
        fontsize=10, color=COLORS["infected"],
        arrowprops=dict(arrowstyle="->", color=COLORS["infected"], lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333",
                  edgecolor=COLORS["infected"], alpha=0.9),
    )

    ax.annotate(
        f"Layered peak: {layered_peak_val:.0f}%\n(day {layered_peak_day:.0f})",
        xy=(layered_peak_day, layered_peak_val),
        xytext=(layered_peak_day + 60, layered_peak_val - 5),
        fontsize=10, color="#E8A838",
        arrowprops=dict(arrowstyle="->", color="#E8A838", lw=1.5),
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333",
                  edgecolor="#E8A838", alpha=0.9),
    )

    # Info box
    info = ("Same disease parameters:\n"
            "\u03b2=0.3, \u03b3=0.1, \u03c9=0.01\n"
            f"N={int(N):,}, {len(homo)} runs each")
    ax.text(0.98, 0.97, info, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", horizontalalignment="right",
            color="#cccccc",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333",
                      edgecolor="#555555", alpha=0.9))

    ax.set_xlabel("Time (days)", fontsize=13)
    ax.set_ylabel("Infected (%)", fontsize=13)
    ax.set_title("Structured Contacts Change Epidemic Dynamics", fontsize=15, pad=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11, facecolor="#2a2a2a", edgecolor="#555555")

    fig.text(0.5, 0.01,
             "Same micro-level SIRS rules \u2014 only contact structure differs",
             ha="center", fontsize=10, color="#999999", style="italic")

    plt.subplots_adjust(bottom=0.1)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def plot_intervention_ranking(structured_data, output_path=None, n_ensemble=None):
    """Plot grouped bar chart comparing interventions across model types.

    Args:
        structured_data: dict with 'layered_results' and 'homogeneous_results'.
        output_path: Path to save figure.
        n_ensemble: Number of runs per condition.

    Returns:
        matplotlib Figure.
    """
    _apply_style()

    layered = structured_data["layered_results"]
    homogeneous = structured_data["homogeneous_results"]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Intervention labels and data
    interventions = ["no_intervention", "random_vax", "targeted_vax",
                     "close_workplace", "close_community"]
    labels = ["No\nIntervention", "Random\nVaccination", "Targeted\n(Hub) Vax",
              "Close\nWorkplaces", "Close\nCommunity"]

    layered_means = [layered[c]["peak_mean"] * 100 for c in interventions]
    layered_stds = [layered[c]["peak_std"] * 100 for c in interventions]

    # Homogeneous only has first 3 conditions
    homo_means = [homogeneous[c]["peak_mean"] * 100 for c in interventions[:3]]
    homo_stds = [homogeneous[c]["peak_std"] * 100 for c in interventions[:3]]

    x = np.arange(len(interventions))
    width = 0.35

    # Homogeneous bars (only first 3)
    bars_homo = ax.bar(x[:3] - width / 2, homo_means, width,
                       yerr=homo_stds, capsize=5,
                       color=COLORS["infected"], alpha=0.8,
                       label="Homogeneous Model", edgecolor="#555555")

    # "Not representable" markers for layer interventions in homogeneous model
    for idx in [3, 4]:
        ax.text(x[idx] - width / 2, 1.0,
                "Not\nrepresentable",
                fontsize=8, color="#999999", ha="center", va="bottom",
                style="italic")

    # Layered bars (all 5)
    bars_layered = ax.bar(x + width / 2, layered_means, width,
                          yerr=layered_stds, capsize=5,
                          color="#E8A838", alpha=0.8,
                          label="Layered Network Model", edgecolor="#555555")

    # Highlight the key finding: targeted > random in layered model
    if len(layered_means) >= 3:
        random_val = layered_means[1]
        targeted_val = layered_means[2]
        if targeted_val < random_val:
            # Draw bracket between random and targeted bars in layered group
            bracket_x1 = 1 + width / 2
            bracket_x2 = 2 + width / 2
            bracket_y = max(random_val, targeted_val) + max(layered_stds[1], layered_stds[2]) + 2
            diff_pct = (random_val - targeted_val) / random_val * 100

            ax.annotate(
                f"Hub targeting: {diff_pct:.0f}% more\neffective than random",
                xy=((bracket_x1 + bracket_x2) / 2, bracket_y),
                xytext=((bracket_x1 + bracket_x2) / 2 + 0.8, bracket_y + 4),
                fontsize=10, color="#66CC99", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#66CC99", lw=1.5),
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333",
                          edgecolor="#66CC99", alpha=0.9),
            )

    # Check if homogeneous model shows no difference between random and targeted
    if len(homo_means) >= 3:
        homo_diff = abs(homo_means[1] - homo_means[2])
        if homo_diff < 3:  # Within 3 percentage points
            bracket_y_homo = max(homo_means[1], homo_means[2]) + max(homo_stds[1], homo_stds[2]) + 2
            ax.text(1.5 - width / 2, bracket_y_homo + 1,
                    "Indistinguishable\nin homogeneous model",
                    fontsize=9, color=COLORS["infected"], ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="#333333",
                              edgecolor=COLORS["infected"], alpha=0.9))

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Peak Infection (% of population)", fontsize=13)
    ax.set_title("Intervention Rankings Change With Model Structure", fontsize=15, pad=10)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=11, facecolor="#2a2a2a", edgecolor="#555555", loc="upper right")

    # Info box
    info = (f"{n_ensemble or '?'} runs per condition\n"
            "Vaccination: 20% of population\n"
            "Layer reduction: 50% of edges")
    ax.text(0.02, 0.97, info, transform=ax.transAxes,
            fontsize=10, verticalalignment="top", color="#cccccc",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333",
                      edgecolor="#555555", alpha=0.9))

    fig.text(0.5, 0.01,
             "The model you choose determines which interventions you can even evaluate",
             ha="center", fontsize=10, color="#999999", style="italic")

    plt.subplots_adjust(bottom=0.1)

    if output_path:
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    return fig


def save_structured_plots(structured_data, output_dir,
                           n_steps=None, dt=None, n_ensemble=None):
    """Generate and save experiment 4 plots.

    Returns:
        List of saved file Paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved = []

    fig = plot_structured_comparison(structured_data,
                                     output_dir / "structured_comparison.png",
                                     n_steps=n_steps, dt=dt)
    plt.close(fig)
    saved.append(output_dir / "structured_comparison.png")

    fig = plot_intervention_ranking(structured_data,
                                    output_dir / "intervention_ranking.png",
                                    n_ensemble=n_ensemble)
    plt.close(fig)
    saved.append(output_dir / "intervention_ranking.png")

    return saved
