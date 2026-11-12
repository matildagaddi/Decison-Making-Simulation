"""Figures for a saved run.

    python -m harvest.experiments.plot [results/<run>]   (default: latest results/baseline_*)

Writes <run>/figures/{reward,reward_distribution,fruit_outcomes,episode}.{png,pdf}.
"""
import argparse
import csv
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# ---- palette (light surface; categorical slots in fixed order) ----
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e1e0d9"
NEUTRAL = "#9b9a94"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "axes.titlecolor": TEXT,
    "axes.titlesize": 12, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "axes.labelsize": 10, "xtick.color": TEXT_2, "ytick.color": TEXT_2,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "legend.frameon": False, "legend.fontsize": 9,
    "text.color": TEXT, "font.size": 10,
})


def read_csv(path):
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(x):
    if x in ("", "None", None):
        return None
    if x in ("True", "False"):
        return x == "True"
    try:
        return float(x)
    except ValueError:
        return x


def save(fig, out, name):
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(out, f"{name}.{ext}"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ---------- 1. mean reward with 95% CI ----------
def plot_reward(summary, out):
    s = sorted(summary, key=lambda r: float(r["reward"]))
    names = [r["policy"] for r in s]
    m = np.array([float(r["reward"]) for r in s])
    ci = np.array([float(r["reward_ci95"]) for r in s])
    y = np.arange(len(s))
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(s) + 1.2))
    ax.errorbar(m, y, xerr=ci, fmt="o", color=SERIES[0], ecolor=SERIES[0],
                elinewidth=2, capsize=4, markersize=7, zorder=3)
    for yi, mi, ci_i in zip(y, m, ci):
        ax.annotate(f"{mi:.1f} ± {ci_i:.1f}", (mi + ci_i, yi), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8.5, color=TEXT_2)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(y, names)
    ax.grid(axis="y", visible=False)
    lo, hi = (m - ci).min(), (m + ci).max()
    ax.set_xlim(min(0, lo) - 5, hi + 0.18 * (hi - min(0, lo)))
    ax.set_xlabel("Mean total reward per session (95% CI)")
    ax.set_title(f"Mean reward by policy (n = {s[0]['n_episodes']} shared scenarios)")
    save(fig, out, "reward")


# ---------- 2. per-episode reward distribution ----------
def plot_distribution(episodes, summary, out):
    order = [r["policy"] for r in sorted(summary, key=lambda r: float(r["reward"]))]
    data = [[float(e["reward"]) for e in episodes if e["policy"] == p] for p in order]
    fig, ax = plt.subplots(figsize=(7, 0.45 * len(order) + 1.2))
    bp = ax.boxplot(data, vert=False, widths=0.55, patch_artist=True, whis=(5, 95),
                    showfliers=True, medianprops=dict(color=TEXT, linewidth=2),
                    flierprops=dict(marker="o", markersize=2.5, markerfacecolor=NEUTRAL,
                                    markeredgecolor="none", alpha=0.6),
                    whiskerprops=dict(color=TEXT_2, linewidth=1),
                    capprops=dict(color=TEXT_2, linewidth=1))
    for box in bp["boxes"]:
        box.set(facecolor=SERIES[0] + "55", edgecolor=SERIES[0], linewidth=1.2)
    ax.axvline(0, color=TEXT_2, linewidth=0.8)
    ax.set_yticks(range(1, len(order) + 1), order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Total reward per session (box: IQR, line: median, whiskers: 5th–95th pct)")
    ax.set_title("Per-episode reward distribution")
    save(fig, out, "reward_distribution")


# ---------- 3. what happened to each fruit ----------
FATES = [("good", "Harvested, good", SERIES[0]),
         ("bad", "Harvested, bad", SERIES[1]),
         ("damaged", "Plant damaged", SERIES[2]),
         ("skipped_all", "Skipped / abandoned", SERIES[3]),
         ("unreached", "Not reached (time ran out)", NEUTRAL)]


def plot_outcomes(episodes, summary, out):
    order = [r["policy"] for r in sorted(summary, key=lambda r: float(r["reward"]))]
    means = {}
    for p in order:
        rows = [e for e in episodes if e["policy"] == p]
        f = {k: np.mean([float(e[k]) for e in rows]) for k in
             ("good", "bad", "damaged", "skipped", "abandoned", "n_fruits")}
        # SKIP is counted per action; a fruit is skipped at most once, so this is per fruit
        f["skipped_all"] = f["skipped"] + f["abandoned"]
        f["unreached"] = f["n_fruits"] - f["good"] - f["bad"] - f["damaged"] - f["skipped_all"]
        means[p] = f
    y = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(order) + 1.6))
    left = np.zeros(len(order))
    for key, label, color in FATES:
        w = np.array([means[p][key] for p in order])
        ax.barh(y, w, left=left, height=0.62, color=color, label=label,
                edgecolor=SURFACE, linewidth=2)
        for yi, li, wi in zip(y, left, w):
            if wi >= 6:
                ax.text(li + wi / 2, yi, f"{wi:.0f}", ha="center", va="center",
                        fontsize=8, color=TEXT)
        left += w
    ax.set_yticks(y, order)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Mean number of fruits per session")
    ax.set_title("What happened to each fruit")
    ax.legend(ncol=3, loc="upper left", bbox_to_anchor=(0, -0.11))
    save(fig, out, "fruit_outcomes")


# ---------- 4. one example episode ----------
EPISODE_POLICIES = ["greedy", "threshold (sensor)", "rate (sensor)"]


def plot_episode(traj, cfg_low, session_time, out, episode=0):
    pols = [p for p in EPISODE_POLICIES if any(t["policy"] == p for t in traj)]
    if not pols:
        return
    fig, axes = plt.subplots(len(pols), 1, figsize=(9, 2.3 * len(pols) + 0.8),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    for ax, p in zip(axes, pols):
        steps = [{k: num(v) for k, v in t.items()} for t in traj
                 if t["policy"] == p and int(t["episode"]) == episode]
        t0 = [session_time - s["time_before"] for s in steps]
        t1 = [session_time - s["time_after"] for s in steps]
        # battery: drop happens over the action, recharge shown as a shaded span
        xs, ys = [0.0], [steps[0]["battery_before"]]
        for s, a, b in zip(steps, t0, t1):
            xs += [a, b]
            ys += [s["battery_before"], s["battery_after"]]
            if s["action"] == "RECHARGE":
                ax.axvspan(a, b, color=GRID, alpha=0.7, linewidth=0, zorder=0)
        ax.plot(xs, ys, color=SERIES[0], linewidth=2, zorder=2)
        ax.axhline(cfg_low, color=TEXT_2, linewidth=0.8, linestyle="--", zorder=1)

        def mark(pred, **kw):
            pts = [(b, s["battery_after"]) for s, b in zip(steps, t1) if pred(s)]
            if pts:
                ax.scatter(*zip(*pts), zorder=3, edgecolors=SURFACE, linewidths=1, **kw)
        mark(lambda s: s["harvested_good"] is True, marker="o", s=34, color=SERIES[0], label="Harvested good")
        mark(lambda s: s["harvested_good"] is False, marker="s", s=30, color=SERIES[1], label="Harvested bad")
        mark(lambda s: s["damaged"] is True, marker="X", s=60, color=SERIES[2], label="Plant damaged")
        if steps[-1]["end_reason"] == "battery":
            ax.annotate("battery dead", (t1[-1], 0), xytext=(-8, 14), textcoords="offset points",
                        ha="right", fontsize=8.5, color=TEXT,
                        arrowprops=dict(arrowstyle="-", color=TEXT_2, linewidth=0.8))
        total = sum(s["reward"] for s in steps)
        ax.set_title(f"{p}  ·  reward {total:.0f}", fontsize=10.5)
        ax.set_ylim(-0.05, 1.08)
        ax.set_ylabel("Battery")
    axes[0].text(2, cfg_low + 0.02, "low-battery threshold", ha="left",
                 va="bottom", fontsize=8, color=TEXT_2)
    axes[-1].set_xlim(0, session_time)
    axes[-1].set_xlabel("Elapsed session time (shaded: recharging)")
    handles, labels = [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels, ncol=3, loc="lower left", bbox_to_anchor=(0.06, -0.04))
    fig.suptitle(f"Example episode {episode}: battery over one session", x=0.06,
                 ha="left", fontsize=12, fontweight="semibold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, out, "episode")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run", nargs="?", help="results/<run> directory")
    ap.add_argument("--episode", type=int, default=0, help="logged episode for the episode plot")
    args = ap.parse_args()
    run = args.run or max(glob.glob("results/baseline_*"), default=None)
    if not run:
        raise SystemExit("no results/baseline_* run found; pass a run directory")
    out = os.path.join(run, "figures")
    os.makedirs(out, exist_ok=True)

    summary = read_csv(os.path.join(run, "summary.csv"))
    episodes = read_csv(os.path.join(run, "episodes.csv"))
    from ..config import Config
    cfg = Config.load(os.path.join(run, "config.json"))

    plot_reward(summary, out)
    plot_distribution(episodes, summary, out)
    plot_outcomes(episodes, summary, out)
    tpath = os.path.join(run, "trajectories.csv")
    if os.path.exists(tpath):
        plot_episode(read_csv(tpath), cfg.low_battery, cfg.session_time, out, args.episode)
    print("Saved figures to", out)


if __name__ == "__main__":
    main()
