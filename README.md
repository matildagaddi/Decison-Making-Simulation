# Decision-Making Simulation
CS238 / AA228 — time and battery allocation for a fruit-harvesting robot.

## Layout
```
harvest/
  config.py            Config dataclass (+ JSON save/load)
  field.py             generate_field(cfg, seed): fruits, seeded separately from outcomes
  env.py               HarvestEnv dynamics: step(), valid_actions(), per-step info dict
  belief.py            BeliefTracker: Beta success belief + Bayes P(good) (supports misspecification)
  policies/            heuristic baselines; every policy is built from the env's Config
  evaluate.py          shared scenarios, paired comparisons, result saving
  experiments/         one script per study
tests/                 sanity tests for the dynamics, beliefs and evaluation
results/<exp>_<stamp>/ config.json, meta.json (git commit, seed), summary/episodes/trajectories CSVs
```

## Run
```
python -m harvest.experiments.baseline --episodes 1000 --seed 0
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

## Model
Actions: HARVEST, SKIP, CHECK (noisy quality sensor), RECHARGE. Fruits are visited in a fixed order.

- Leaving a fruit (harvest success, damage, abandon, skip) costs `move_time` and `move_cost`.
- An action is valid only if its worst-case duration fits in the remaining session time.
- Discounting is per time unit (`gamma ** duration`); the low-battery penalty is per time unit.
- Damage probability on a failed attempt is `min(1, damage_base * failures)`; `max_attempts` optionally abandons a fruit.
- Sensor readings: `sens_tp` / `sens_fp`; with `sens_rho > 0` each repeat reading copies the first one w.p. rho.
- Every scenario has its own field seed, and each fruit gets its own harvest and sensor RNG, so policies face the same fields and the same per-fruit outcomes (common random numbers). Comparisons use paired differences.

## Known limitations / future work
- **Spatial structure (recommended next model extension).** Plants currently only set fruit counts. Adding travel between plants, a dock location, and plant-level damage (losing the plant's remaining fruit) would make routing and recharge timing real decisions, at the cost of a much larger state space.
- No idle/stop action: near the end of a session with too little time to recharge, every action drains battery.
- Quality, difficulty and success probability are independent; difficulty and battery are observed exactly.
- Baselines are heuristics; planners (oracle bound, belief-MDP value iteration, MCTS/POMCP) are the next phase.

`results/legacy_20261110_112751/` holds the output of the original single-file sim (it was run with `recharge_time=6` and per-action discounting; not comparable with current results).
