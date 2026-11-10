"""Evaluate the heuristic baselines on a shared scenario set.

    python -m harvest.experiments.baseline [--episodes 1000] [--seed 0] [--log 5] [--config path.json]
                                           [--set key=value ...] [--name baseline]

--set overrides Config fields; values are parsed as JSON, e.g.
    --set weather=true --set weather_observed=false --set allow_wait=true
"""
import argparse
import json

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
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                    help="override a Config field (value parsed as JSON)")
    ap.add_argument("--name", default="baseline", help="results folder prefix")
    args = ap.parse_args()

    cfg = Config.load(args.config) if args.config else Config()
    if args.set:
        d = cfg.to_dict()
        for kv in args.set:
            k, v = kv.split("=", 1)
            if k not in d:
                ap.error(f"unknown Config field: {k}")
            try:
                d[k] = json.loads(v)
            except json.JSONDecodeError:
                d[k] = v
        cfg = Config.from_dict(d)
    scenarios = make_scenarios(args.episodes, args.seed)
    rows, traj = evaluate(baseline_policies(cfg), cfg, scenarios, n_log=args.log)
    summary = summarize(rows)
    print_summary(summary)
    if not args.no_save:
        print("\nSaved:", save_run(args.name, cfg, args.seed, summary, rows, traj, args.out))


if __name__ == "__main__":
    main()
