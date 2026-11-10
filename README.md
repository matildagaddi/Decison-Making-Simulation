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

So far we've tried simple rules of thumb:
- **Greedy:** pick everything, never give up. It does terribly: it wastes time on rotten fruit
  and runs its battery dead.
- **Threshold:** skip fruit that looks bad or too hard. It does much better.
- **Rate-based:** only pick a fruit if it's worth the time. It's the best so far.

One clear finding is that **the sensor is worth using**. Checking fruit before picking adds
about 13 points per session.

### What's next

1. Build a **"cheating" robot** that knows everything: which fruits are good and how hard
   each is to pick. It shows the best score that's even possible.
2. Build a robot that **plans ahead** using decision-making methods from CS238, like value
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
