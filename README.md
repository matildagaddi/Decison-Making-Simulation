# Decision-Making Simulation
CS238 / AA228 — time and battery allocation for a fruit-harvesting robot.

## What we're trying to do (in plain words)

Imagine a robot picking fruit in a field. It has **limited time** (one work session) and a
**limited battery**. Its goal is to collect as many good fruits as possible without wasting
time, draining its battery or damaging plants.

The robot goes down a row of fruit, one at a time. At each fruit, it chooses between four things:

- **Pick it.** This might fail, and if it keeps failing, the robot might damage the plant.
- **Skip it** and move on.
- **Check it** with a sensor to see whether it's good or rotten. The sensor isn't always right.
- **Recharge** its battery. This takes a long time.

### Why it's tricky

The robot **can't see everything**. It doesn't know for sure whether a fruit is good, or how
hard it will be to pick. It only has guesses, and the guesses get better as it checks the
fruit or tries and fails to pick it.

Every choice has a trade-off:
- Checking costs time, but helps avoid picking rotten fruit.
- Retrying a hard fruit might work, or it might waste time and damage the plant.
- Recharging protects the battery, but uses up time that could go to picking.

### Rewards and penalties

| Event | Points |
|---|---|
| Pick a good fruit | +2 |
| Pick a rotten fruit | −1 |
| Damage a plant | −2 |
| Each time unit with a nearly empty battery | −2 |
| Battery dies completely | −10 |

### What we're trying to find out

**What's the smartest strategy for the robot?** We test different strategies, or "policies,"
in a simulation, run each one a thousand times, and compare their scores.

### Results so far

The robot has 200 time units to work through a field of about 100 fruits. Every strategy is
tested on the **same 1000 randomly generated fields with the same luck**, so differences in
score come from the strategy and not from chance.

#### Without weather

| Strategy | What it does | Avg. score |
|---|---|---|
| **Rate-based** (with sensor) | Picks a fruit only if it's worth the time | **40.1** |
| Threshold, strict (with sensor) | Skips fruit that looks bad or hard, with stricter cut-offs | 36.9 |
| Threshold (with sensor) | Skips fruit that looks bad or hard | 34.5 |
| Threshold (no sensor) | Same, but never checks fruit quality | 20.9 |
| Random | Picks actions at random | 5.7 |
| **Greedy** | Picks everything and never gives up | **−31.3** |

- **Being selective wins.** In a typical session, the best strategy picks about 22 good fruits
  and only 2–3 rotten ones, and damages a plant less than once. It skips about half the fruit,
  because spending time on unpromising fruit isn't worth it.
- **Greedy is a disaster.** It picks about as many good fruits as the best strategy (21), but
  also about 14 rotten ones, damages about 5 plants, and drains its battery completely in about
  29% of sessions.
- **The quality sensor is worth using.** With the sensor, the threshold strategy scores about
  13.5 points more. Without it, the strategy picks slightly *more* good fruit but also 15
  rotten ones, because it can't tell them apart.
- **Fine-tuning barely matters.** A "pickier" version of the rate-based strategy scores the same
  (40.1). What matters is the overall approach, not the exact cut-off.

#### With weather (preliminary)

Rain and heat make picking harder, and rain makes damaging plants more likely. We tried two
versions: one where the robot can see the weather, and one where it has to spend time checking.

| Strategy | Can see weather | Hidden weather |
|---|---|---|
| Rate-based | 36.2 | 36.2 |
| Rate-based, waits out bad weather | 36.2 | 32.7* |
| Rate-based, checks weather every 10 time units | not applicable | 32.7 |
| Greedy | −35.7 | −35.7 |

\* Because of a bug, this strategy never actually waits. It ends up behaving exactly like the
"checks weather" strategy.

- **Bad weather costs about 4 points per session** (40 → 36). Fewer picks succeed, so the robot
  picks about 1–2 fewer good fruits and damages a few more plants.
- **Knowing the weather doesn't help yet.** The rate-based strategy scores the same whether it
  can see the weather or not.
- **Checking the weather costs more than it's worth.** Checking every 10 time units uses up time
  that could go to picking, and scores about 3.6 points worse.
- **Waiting out bad weather barely helps** (+0.002 points). Weather only lasts about 10 time
  units, and the robot usually has another fruit to work on.

**The big picture:** simple rules already do well, and the rate-based rule is the best so far.
But we don't know yet how good "well" is. That's what the next steps are for.

Full numbers: `results/baseline_20261110_121831/summary.csv`,
`results/weather_observed_20261110_124410/summary.csv`,
`results/weather_hidden_20261110_124550/summary.csv`.

### What's next

1. Fix the "wait out bad weather" bug and re-run the weather experiments.
2. Build a **"cheating" robot** that knows everything: which fruits are good and how hard
   each is to pick. It shows the best score that's even possible.
3. Build a robot that **plans ahead** using decision-making methods from CS238, like value
   iteration or tree search.

Then we can see how close our strategies get to the best possible score, and whether careful
planning beats simple rules.

**In one sentence:** we're teaching a robot to make good choices when it has limited time
and battery and can't be sure what it's looking at.

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
# weather (observed / hidden) and the WAIT action are opt-in Config fields:
python -m harvest.experiments.baseline --set weather=true --set allow_wait=true --name weather_observed
python -m harvest.experiments.baseline --set weather=true --set weather_observed=false --set allow_wait=true --name weather_hidden
python -m harvest.experiments.plot [results/<run>]   # figures -> <run>/figures (default: latest baseline)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q tests
```

## Model
Actions: HARVEST, SKIP, CHECK (noisy quality sensor), RECHARGE. Fruits are visited in a fixed order.

- Leaving a fruit (harvest success, damage, abandon, skip) costs `move_time` and `move_cost`.
- An action is valid only if its worst-case duration fits in the remaining session time.
- Discounting is per time unit (`gamma ** duration`); the low-battery penalty is per time unit.
- Damage probability on a failed attempt is `min(1, damage_base * failures)`; `max_attempts` optionally abandons a fruit.
- Sensor readings: `sens_tp` / `sens_fp`; with `sens_rho > 0` each repeat reading copies the first one w.p. rho.
- **Weather (opt-in, `weather=True`).** Markov chain over normal/rain/heat that steps every `weather_period` time units (proposal transition matrix by default). Success is `weather_success_mult[w] * theta`; damage probability is scaled by `weather_damage_mult[w]`. The weather path depends only on elapsed time and the outcome seed, so it is shared across policies.
  - `weather_observed=True`: the robot sees the weather; failures are counted per weather type, which is a sufficient statistic for the success posterior `Beta(a,b) * prod_w (1 - m_w theta)^k_w`.
  - `weather_observed=False`: weather is hidden; `CHECK_WEATHER` returns a reading that is correct w.p. `weather_sens_acc`. `BeliefTracker.p_weather` is an exact HMM filter over those readings (it ignores the weak evidence from harvest outcomes), and `p_success` uses the belief-mean multiplier (approximation).
- **WAIT (opt-in, `allow_wait=True`)** idles for `wait_time` with no battery cost (low-battery penalty still applies).
- Every scenario has its own field seed, and each fruit gets its own harvest and sensor RNG, so policies face the same fields and the same per-fruit outcomes (common random numbers). Comparisons use paired differences.

## Known limitations / future work
- **Spatial structure (recommended next model extension).** Plants currently only set fruit counts. Adding travel between plants, a dock location, and plant-level damage (losing the plant's remaining fruit) would make routing and recharge timing real decisions, at the cost of a much larger state space.
- No idle/stop action unless `allow_wait=True` (default off to keep earlier results reproducible).
- `prior_beta` differs from `true_beta` by default, so the agent's success belief is not the true posterior. Set them equal for Bayes-optimal benchmarks.
- Quality, difficulty and success probability are independent; difficulty and battery are observed exactly.
- Baselines are heuristics; planners (oracle bound, belief-MDP value iteration, MCTS/POMCP) are the next phase.

`results/legacy_20261110_112751/` holds the output of the original single-file sim (it was run with `recharge_time=6` and per-action discounting; not comparable with current results).
