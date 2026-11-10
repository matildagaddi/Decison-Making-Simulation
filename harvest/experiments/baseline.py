"""Evaluate the heuristic baselines on a shared scenario set.

    python -m harvest.experiments.baseline [--episodes 1000] [--seed 0] [--log 5] [--config path.json]
"""
import argparse

from ..config import Config
from ..evaluate import evaluate, make_scenarios, print_summary, save_run, summarize
from ..policies import baseline_policies


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log", type=int, default=5, help="episodes per policy to save step-by-step")
    ap.add_argument("--config", help="JSON config (defaults to Config())")
    ap.add_argument("--out", default="results")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    scenarios = make_scenarios(args.episodes, args.seed)
    rows, traj = evaluate(baseline_policies(), cfg, scenarios, n_log=args.log)
    summary = summarize(rows)
    print_summary(summary)
    if not args.no_save:
        print("\nSaved:", save_run("baseline", cfg, args.seed, summary, rows, traj, args.out))


if __name__ == "__main__":
    main()
