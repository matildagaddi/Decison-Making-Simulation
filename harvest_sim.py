"""
Barebones simulation: time & battery allocation for a fruit-harvesting robot.

Field: 20 plants, ~5 fruits each. Each fruit has a hidden quality (good/bad) and a
hidden success probability for a harvest attempt, determined by its difficulty class.
The robot works through fruits in order and, for the current fruit, chooses among:

    HARVEST   - attempt (or retry) picking the current fruit
    SKIP      - move on to the next fruit without harvesting
    CHECK     - use the quality sensor (noisy, small battery cost)
    RECHARGE  - dock and charge (takes several time units, so it is costly in time)

Each session has a fixed time budget (cfg.session_time). Every action takes time;
if an action does not fit in the remaining time, the session ends with no effect.

Rewards (from the proposal):
    good fruit harvested +2 | bad fruit harvested -1 | plant damaged -2
    low battery (< LOW_BATTERY) -2 per step | battery runs out -10 (episode ends)

Beliefs the agent keeps (and can use in a policy):
    * success-probability belief Beta(a, b), initialised from a difficulty-class prior;
      b += 1 after every failed attempt (so repeated failures lower the estimate)
    * P(fruit is good), updated by Bayes' rule after each CHECK reading

Run:  python harvest_sim.py
"""
from dataclasses import dataclass, field
from enum import IntEnum
import numpy as np
from datetime import datetime


class Action(IntEnum):
    HARVEST = 0
    SKIP = 1
    CHECK = 2
    RECHARGE = 3


EASY, HARD = 0, 1


@dataclass
class Config:
    n_plants: int = 20
    mean_fruits_per_plant: float = 5.0
    session_time: int = 200         # time budget per harvesting session (time units)
    gamma: float = 0.99             # discount used for reported return

    # fruit generation
    p_hard: float = 0.4
    p_good: float = 0.6
    # true success prob ~ Beta(*) by difficulty
    true_beta: dict = field(default_factory=lambda: {EASY: (8, 2), HARD: (3, 5)})
    # agent's prior belief over success prob (Beta params) by difficulty
    prior_beta: dict = field(default_factory=lambda: {EASY: (4, 1), HARD: (2, 3)})

    # battery
    harvest_cost: dict = field(default_factory=lambda: {EASY: 0.02, HARD: 0.06})
    sensor_cost: float = 0.005
    move_cost: float = 0.005
    recharge_rate: float = 0.30     # battery gained per RECHARGE action

    # action durations (time units)
    harvest_time: dict = field(default_factory=lambda: {EASY: 1, HARD: 2})
    check_time: int = 1
    move_time: int = 1
    recharge_time: int = 10 
    low_battery: float = 0.10

    # damage on a failed attempt: prob = base * (number of failed attempts so far)
    damage_base: dict = field(default_factory=lambda: {EASY: 0.05, HARD: 0.10})

    # sensor: P(reads "good" | good), P(reads "good" | bad)
    sens_tp: float = 0.85
    sens_fp: float = 0.15

    # rewards
    r_good: float = 2.0
    r_bad: float = -1.0
    r_damage: float = -2.0
    r_low: float = -2.0
    r_dead: float = -10.0


@dataclass
class Fruit:
    difficulty: int
    good: bool
    true_p: float


class HarvestEnv:
    def __init__(self, cfg: Config = None, seed: int = None):
        self.cfg = cfg or Config()
        self.rng = np.random.default_rng(seed)

    # ---------- setup ----------
    def reset(self):
        c, rng = self.cfg, self.rng
        self.fruits = []
        for _ in range(c.n_plants):
            for _ in range(max(1, rng.poisson(c.mean_fruits_per_plant))):
                d = HARD if rng.random() < c.p_hard else EASY
                a, b = c.true_beta[d]
                self.fruits.append(Fruit(d, rng.random() < c.p_good, rng.beta(a, b)))
        self.idx = 0
        self.battery = 1.0
        self.t = 0                      # number of actions taken
        self.time_left = c.session_time
        self.done = False
        self.stats = dict(good=0, bad=0, damaged=0, dead=0, skipped=0, recharges=0)
        self._new_fruit_beliefs()
        return self.observe()

    def _new_fruit_beliefs(self):
        if self.idx < len(self.fruits):
            self.attempts = 0
            self.beta_ab = list(self.cfg.prior_beta[self.fruits[self.idx].difficulty])
            self.p_good_belief = self.cfg.p_good

    # ---------- observation (what the agent sees) ----------
    def observe(self):
        a, b = self.beta_ab if self.idx < len(self.fruits) else (1, 1)
        return dict(
            battery=self.battery,
            difficulty=self.fruits[self.idx].difficulty if self.idx < len(self.fruits) else None,
            attempts=self.attempts if self.idx < len(self.fruits) else 0,
            p_success=a / (a + b),
            p_good=self.p_good_belief if self.idx < len(self.fruits) else None,
            fruits_left=len(self.fruits) - self.idx,
            time_left=self.time_left,
        )

    # ---------- dynamics ----------
    def _advance(self):
        self.idx += 1
        if self.idx >= len(self.fruits):
            self.done = True
        else:
            self._new_fruit_beliefs()

    def step(self, action):
        c, rng = self.cfg, self.rng
        assert not self.done
        f = self.fruits[self.idx]
        r = 0.0

        dur = self.duration(action, f)
        if dur > self.time_left:                 # session time expires mid-action
            self.time_left = 0
            self.done = True
            return self.observe(), 0.0, True
        self.time_left -= dur

        if action == Action.HARVEST:
            self.battery -= c.harvest_cost[f.difficulty]
            if self.battery > 0:
                if rng.random() < f.true_p:                      # success
                    if f.good:
                        r += c.r_good; self.stats["good"] += 1
                    else:
                        r += c.r_bad; self.stats["bad"] += 1
                    self._advance()
                else:                                            # failure
                    self.attempts += 1
                    self.beta_ab[1] += 1                         # belief update
                    if rng.random() < c.damage_base[f.difficulty] * self.attempts:
                        r += c.r_damage; self.stats["damaged"] += 1
                        self._advance()                          # plant damaged: fruit lost

        elif action == Action.SKIP:
            self.battery -= c.move_cost
            self.stats["skipped"] += 1
            if self.battery > 0:
                self._advance()

        elif action == Action.CHECK:
            self.battery -= c.sensor_cost
            if self.battery > 0:
                p_read_good = c.sens_tp if f.good else c.sens_fp
                reads_good = rng.random() < p_read_good
                # Bayes update of P(good)
                pg = self.p_good_belief
                like_g = c.sens_tp if reads_good else 1 - c.sens_tp
                like_b = c.sens_fp if reads_good else 1 - c.sens_fp
                self.p_good_belief = pg * like_g / (pg * like_g + (1 - pg) * like_b)

        elif action == Action.RECHARGE:
            self.battery = min(1.0, self.battery + c.recharge_rate)
            self.stats["recharges"] += 1

        # battery penalties
        if self.battery <= 0:
            self.battery = 0.0
            r += c.r_dead; self.stats["dead"] = 1; self.done = True
        elif self.battery < c.low_battery:
            r += c.r_low

        self.t += 1
        if self.time_left <= 0:
            self.done = True
        return self.observe(), r, self.done

    def duration(self, action, fruit):
        c = self.cfg
        return {Action.HARVEST: c.harvest_time[fruit.difficulty],
                Action.SKIP: c.move_time,
                Action.CHECK: c.check_time,
                Action.RECHARGE: c.recharge_time}[action]


# ---------------- baseline policies ----------------
def random_policy(obs, rng):
    return Action(rng.integers(0, 4))


def greedy_policy(obs, rng):
    """Harvest everything, retry until success, recharge only when nearly empty."""
    if obs["battery"] < 0.08:
        return Action.RECHARGE
    return Action.HARVEST


def make_threshold_policy(recharge_below=0.25, min_p_good=0.5, min_p_success=0.3,
                          use_sensor=True, max_checks_per_fruit=1):
    """Less-greedy: recharge below a battery threshold; optionally check quality once;
    skip fruits that look bad or whose success belief has dropped below a threshold."""
    state = {"idx_checked": None, "n_checks": 0}

    def policy(obs, rng):
        # only recharge if there is time to finish a recharge and still do something useful
        if obs["battery"] < recharge_below and obs["time_left"] > Config().recharge_time + 2:
            return Action.RECHARGE
        key = obs["fruits_left"]
        if state["idx_checked"] != key:                          # new fruit
            state["idx_checked"], state["n_checks"] = key, 0
        if use_sensor and obs["attempts"] == 0 and state["n_checks"] < max_checks_per_fruit:
            state["n_checks"] += 1
            return Action.CHECK
        if obs["p_good"] < min_p_good or obs["p_success"] < min_p_success:
            return Action.SKIP
        return Action.HARVEST

    return policy


def make_rate_policy(recharge_below=0.25, min_rate=0.5, use_sensor=True):
    """Time-aware: harvest only if expected reward per unit time >= min_rate."""
    cfg = Config()
    state = {"key": None, "checked": False}

    def policy(obs, rng):
        if obs["battery"] < recharge_below and obs["time_left"] > cfg.recharge_time + 2:
            return Action.RECHARGE
        if state["key"] != obs["fruits_left"]:
            state["key"], state["checked"] = obs["fruits_left"], False
        if use_sensor and obs["attempts"] == 0 and not state["checked"]:
            state["checked"] = True
            return Action.CHECK
        pg = obs["p_good"]
        exp_r = obs["p_success"] * (pg * cfg.r_good + (1 - pg) * cfg.r_bad)
        rate = exp_r / cfg.harvest_time[obs["difficulty"]]
        return Action.HARVEST if rate >= min_rate else Action.SKIP

    return policy


# ---------------- evaluation ----------------
def run_episode(env, policy, policy_rng):
    obs, total, disc = env.reset(), 0.0, 1.0
    ret = 0.0
    while not env.done:
        obs, r, _ = env.step(policy(obs, policy_rng))
        total += r
        ret += disc * r
        disc *= env.cfg.gamma
    return total, ret, dict(env.stats), env.t


def evaluate(name, make_policy, n_episodes=300, seed=0):
    env = HarvestEnv(seed=seed)
    prng = np.random.default_rng(seed + 1)
    rows = []
    for _ in range(n_episodes):
        total, disc_ret, st, T = run_episode(env, make_policy(), prng)
        rows.append([total, disc_ret, st["good"], st["bad"], st["damaged"], st["dead"], T])
    m, se = np.mean(rows, 0), np.std(rows, 0) / np.sqrt(n_episodes)
    print(f"{name:<28} reward {m[0]:7.1f} ±{se[0]:4.1f} | disc {m[1]:6.1f} | "
          f"good {m[2]:5.1f} bad {m[3]:4.1f} dmg {m[4]:4.1f} dead {m[5]:4.2f} | actions {m[6]:5.0f}")
    return dict(policy=name, n_episodes=n_episodes, seed=seed,
                reward_mean=m[0], reward_se=se[0], discounted_return=m[1],
                good=m[2], bad=m[3], damaged=m[4], dead_rate=m[5], actions=m[6])


def save_results(results, cfg, out_dir="."):
    """Write results to results_YYYYMMDD_HHMMSS.csv (config saved alongside as .json)."""
    import csv, json, os
    from dataclasses import asdict
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"results_{stamp}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        w.writeheader()
        w.writerows(results)
    cfg_path = os.path.join(out_dir, f"config_{stamp}.json")
    with open(cfg_path, "w") as f:
        json.dump(asdict(cfg), f, indent=2, default=str)
    print(f"\nSaved: {csv_path}\n       {cfg_path}")
    return csv_path


if __name__ == "__main__":
    print(f"{'policy':<28} (mean over episodes)")
    results = [
        evaluate("random", lambda: random_policy),
        evaluate("greedy", lambda: greedy_policy),
        evaluate("threshold (no sensor)", lambda: make_threshold_policy(use_sensor=False)),
        evaluate("threshold (sensor)", lambda: make_threshold_policy(use_sensor=True)),
        evaluate("rate-based (sensor)", lambda: make_rate_policy()),
        evaluate("rate-based (sensor, picky)", lambda: make_rate_policy(min_rate=0.8)),
        evaluate("threshold (sensor, strict)",
                 lambda: make_threshold_policy(recharge_below=0.3, min_p_good=0.7, min_p_success=0.4)),
    ]
    save_results(results, Config())