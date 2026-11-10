# Changelog

What changed on the `rebuild-sim` branch compared to `main`, in plain terms.

---

## Unreleased: weather and a "wait" option (not committed yet)

### What's new

- **Weather.** The field can now have weather: **normal**, **rain** or **heat**. It changes
  every 10 time units, following the transition matrix from our proposal.
  - Rain makes picking harder (success chance ×0.6) and makes plant damage more likely (×1.5).
  - Heat makes picking a bit harder (success chance ×0.85).
  - Every strategy sees the same weather in a given test run, so comparisons stay fair.
- **Two weather modes:**
  - **Weather you can see:** the robot always knows the current weather.
  - **Hidden weather:** the robot has to use a new **Check weather** action to find out, which
    costs a little time and battery.
- **A "wait" action.** The robot can stand still for one time unit without using battery, for
  example to wait for the rain to stop.
- **Smarter guessing.** When a pick fails in the rain, the robot now blames the rain partly,
  not just the fruit. With hidden weather, it keeps a running guess of what the weather
  probably is.
- **New strategies** that use the weather: one that checks the weather every 10 time units,
  and one that waits when the weather is bad.
- **Easier experiments.** You can now change any setting from the command line, for example:
  ```
  python -m harvest.experiments.baseline --set weather=true --set allow_wait=true --name weather_observed
  ```

### Nothing old breaks

Weather and waiting are **off by default**. We re-ran the original experiment with the new
code and got exactly the same scores as before.

### First results (1000 runs each)

| Setup | Best score |
|---|---|
| No weather | about 40 points |
| Weather you can see | about 36 points |
| Hidden weather | about 36 points |

Weather costs the robot about 4 points per session. Without more tuning, the
weather-aware strategies don't beat the plain strategy yet:

- Checking the weather every 10 time units costs more than it's worth (about 3.6 points worse).
- Waiting out bad weather barely changes anything when the robot can see the weather.
- **Bug to fix:** with hidden weather, the "wait out bad weather" strategy never actually waits.
  Its results are identical to the "check weather" strategy.

---

## Rebuild of the simulation (committed)

### Reorganized the code

`main` has one big file, `harvest_sim.py`. We split it into a `harvest/` folder with one
file per job:

| File | What it does |
|---|---|
| `config.py` | All the settings (rewards, battery costs, times...) |
| `field.py` | Creates the plants and fruits |
| `env.py` | The simulation itself: what happens when the robot takes an action |
| `belief.py` | The robot's guesses (is this fruit good? will this pick work?) |
| `policies/` | The strategies (greedy, threshold, rate-based, random) |
| `evaluate.py` | Runs the strategies many times and compares them |
| `experiments/` | Scripts that run an experiment and make plots |

### Fixed mistakes in how the simulation worked

- **Moving costs something.** Moving to the next fruit now always uses time and battery.
- **Penalties count time, not actions.** The low-battery penalty and the discount now apply per
  unit of time. Before, they applied per action, so a long recharge counted the same as a
  quick sensor check.
- **Damage chance can't go over 100%.** Before, after many failed picks it could.
- **No more surprise endings.** If an action doesn't fit in the time left, the robot just can't
  choose it. Before, choosing it ended the session.

### Made comparisons fair

- Every strategy is tested on **the same 1000 fields with the same luck**: the same fruit, and
  the same pick successes and sensor readings. Differences in score come from the strategy,
  not from luck.
- Results include **error bars** (95% confidence intervals), so we can tell real differences
  from noise.

### Other improvements

- Optional settings: give up on a fruit after a set number of tries, or a sensor whose repeat
  readings aren't independent.
- The robot's guesses live in their own file, so we can test what happens when the robot's
  idea of the world is wrong.
- Every run saves its exact settings and the code version, so any result can be reproduced.
- Added automated tests (`tests/test_sim.py`) that check the simulation behaves as expected.
- Old results were moved to `results/legacy_20261110_112751/`.

### Results and plots

- **Baseline results:** 1000 runs per strategy, saved in `results/baseline_20261110_121831/`.
  - **Rate-based** is best (about 40 points).
  - **Threshold** comes next (about 34–37 points).
  - **Greedy** is worst (about −31 points). It picks everything, wastes time on rotten fruit and
    sometimes drains its battery completely.
  - **Using the sensor is worth it:** about 13 points more per session.
- **Plots:** `python -m harvest.experiments.plot` draws charts of scores and fruit outcomes, and
  a step-by-step view of one example session.

---

## What's next

1. Fix the "wait" bug and re-run the weather experiments.
2. Build a **"cheating" robot** that knows everything, to show the best possible score.
3. Build a robot that **plans ahead** with CS238 methods (value iteration, tree search).
