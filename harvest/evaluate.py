"""Evaluation on a shared set of scenarios with paired comparisons."""
import csv
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from .config import Config
from .env import HarvestEnv

METRICS = ["reward", "discounted", "good", "bad", "damaged", "abandoned", "skipped",
           "checks", "recharges", "dead", "weather_checks", "waits", "actions", "time_used"]


@dataclass(frozen=True)
class Scenario:
    index: int
    field_seed: int
    outcome_seed: int
    policy_seed: int


def make_scenarios(n, seed=0):
    """n independent scenarios. Every policy is evaluated on the same list."""
    out = []
    for i, child in enumerate(np.random.SeedSequence(seed).spawn(n)):
        f, o, p = (int(x) for x in child.generate_state(3))
        out.append(Scenario(i, f, o, p))
    return out


def run_episode(env, policy, sc: Scenario, log=None):
    obs = env.reset(sc.field_seed, sc.outcome_seed)
    policy.reset(np.random.default_rng(sc.policy_seed))
    total, disc_ret, disc = 0.0, 0.0, 1.0
    while not env.done:
        obs, r, _, info = env.step(policy.act(obs, env.valid_actions()))
        total += r
        disc_ret += disc * r
        disc *= env.cfg.gamma ** info["duration"]
        if log is not None:
            log.append(dict(episode=sc.index, **info))
    return dict(episode=sc.index, reward=total, discounted=disc_ret, **env.stats,
                actions=env.t, time_used=env.cfg.session_time - env.time_left,
                n_fruits=len(env.fruits), end_reason=env.end_reason)


def evaluate(policies, cfg: Config, scenarios, n_log=0):
    """policies: name -> factory(cfg). Returns (episode rows, trajectory rows)."""
    env = HarvestEnv(cfg)
    rows, traj = [], []
    for name, factory in policies.items():
        for sc in scenarios:
            log = [] if sc.index < n_log else None
            row = run_episode(env, factory(cfg), sc, log)
            rows.append(dict(policy=name, **row))
            if log:
                traj.extend(dict(policy=name, **step) for step in log)
    return rows, traj


def summarize(rows, reference=None):
    """Per-policy means with 95% CIs, plus paired difference in reward vs `reference`
    (default: the policy with the highest mean reward)."""
    names = list(dict.fromkeys(r["policy"] for r in rows))
    by = {n: sorted((r for r in rows if r["policy"] == n), key=lambda r: r["episode"]) for n in names}
    means = {n: np.mean([r["reward"] for r in by[n]]) for n in names}
    reference = reference or max(means, key=means.get)
    ref_reward = np.array([r["reward"] for r in by[reference]])
    out = []
    for n in names:
        x = {m: np.array([r[m] for r in by[n]], dtype=float) for m in METRICS}
        k = len(x["reward"])
        s = dict(policy=n, n_episodes=k)
        for m in METRICS:
            s[m] = x[m].mean()
        s["reward_ci95"] = 1.96 * x["reward"].std(ddof=1) / np.sqrt(k)
        d = x["reward"] - ref_reward
        s["reference"] = reference
        s["diff_vs_ref"] = d.mean()
        s["diff_ci95"] = 1.96 * d.std(ddof=1) / np.sqrt(k)
        out.append(s)
    return sorted(out, key=lambda s: -s["reward"])


def print_summary(summary):
    ref = summary[0]["reference"]
    print(f"{'policy':<34}{'reward':>16}{'Δ vs ' + ref[:14]:>24}  "
          f"{'good':>5} {'bad':>5} {'dmg':>5} {'dead':>5} {'chk':>5} {'rech':>5} {'time':>5}")
    for s in summary:
        print(f"{s['policy']:<34}{s['reward']:8.2f} ±{s['reward_ci95']:5.2f}"
              f"{s['diff_vs_ref']:15.2f} ±{s['diff_ci95']:5.2f}  "
              f"{s['good']:5.1f} {s['bad']:5.1f} {s['damaged']:5.1f} {s['dead']:5.3f} "
              f"{s['checks']:5.1f} {s['recharges']:5.1f} {s['time_used']:5.0f}")


def _write_csv(path, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def _git_info():
    def run(*args):
        try:
            return subprocess.run(["git", *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except Exception:
            return None
    return dict(commit=run("rev-parse", "HEAD"), dirty=bool(run("status", "--porcelain")))


def save_run(name, cfg, scenarios_seed, summary, rows, traj, out_root="results"):
    """results/<name>_<stamp>/{config.json, meta.json, summary.csv, episodes.csv, trajectories.csv}"""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(out_root, f"{name}_{stamp}")
    os.makedirs(out, exist_ok=True)
    cfg.save(os.path.join(out, "config.json"))
    with open(os.path.join(out, "meta.json"), "w") as fh:
        json.dump(dict(experiment=name, timestamp=stamp, scenario_seed=scenarios_seed,
                       n_episodes=summary[0]["n_episodes"], argv=sys.argv,
                       numpy=np.__version__, python=sys.version.split()[0], git=_git_info()),
                  fh, indent=2)
    _write_csv(os.path.join(out, "summary.csv"), summary)
    _write_csv(os.path.join(out, "episodes.csv"), rows)
    _write_csv(os.path.join(out, "trajectories.csv"), traj)
    return out
