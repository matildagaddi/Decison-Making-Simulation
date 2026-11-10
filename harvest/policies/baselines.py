"""Heuristic baselines (ported from the original harvest_sim.py)."""
from ..env import Action
from .base import Policy


class RandomPolicy(Policy):
    name = "random"

    def act(self, obs, valid):
        return valid[self.rng.integers(len(valid))]


class GreedyPolicy(Policy):
    """Harvest everything, retry until success, recharge only when nearly empty."""
    name = "greedy"

    def __init__(self, cfg, recharge_below=0.08, **kw):
        super().__init__(cfg, **kw)
        self.recharge_below = recharge_below

    def act(self, obs, valid):
        if obs["battery"] < self.recharge_below and Action.RECHARGE in valid:
            return Action.RECHARGE
        return self.fallback(Action.HARVEST, valid)


class ThresholdPolicy(Policy):
    """Recharge below a battery threshold; optionally check quality; skip fruits whose
    P(good) or success belief is below a threshold."""
    name = "threshold"

    def __init__(self, cfg, recharge_below=0.25, min_p_good=0.5, min_p_success=0.3,
                 use_sensor=True, max_checks=1, guard=True, **kw):
        super().__init__(cfg, **kw)
        self.recharge_below, self.min_p_good, self.min_p_success = recharge_below, min_p_good, min_p_success
        self.use_sensor, self.max_checks, self.guard = use_sensor, max_checks, guard

    def act(self, obs, valid):
        if (obs["battery"] < self.recharge_below and Action.RECHARGE in valid
                and obs["time_left"] > self.cfg.recharge_time + 2):
            return Action.RECHARGE
        b = self.belief(obs)
        if self.use_sensor and obs["attempts"] == 0 and len(obs["readings"]) < self.max_checks:
            a = Action.CHECK
        elif b["p_good"] < self.min_p_good or b["p_success"] < self.min_p_success:
            a = Action.SKIP
        else:
            a = Action.HARVEST
        a = self.fallback(a, valid)
        return self.battery_guard(obs, a, valid) if self.guard else a


class RatePolicy(Policy):
    """Harvest only if immediate expected reward per unit harvest time >= min_rate."""
    name = "rate"

    def __init__(self, cfg, recharge_below=0.25, min_rate=0.5, use_sensor=True, guard=True, **kw):
        super().__init__(cfg, **kw)
        self.recharge_below, self.min_rate = recharge_below, min_rate
        self.use_sensor, self.guard = use_sensor, guard

    def act(self, obs, valid):
        c = self.cfg
        if (obs["battery"] < self.recharge_below and Action.RECHARGE in valid
                and obs["time_left"] > c.recharge_time + 2):
            return Action.RECHARGE
        if self.use_sensor and obs["attempts"] == 0 and not obs["readings"]:
            a = Action.CHECK
        else:
            b = self.belief(obs)
            pg = b["p_good"]
            exp_r = b["p_success"] * (pg * c.r_good + (1 - pg) * c.r_bad)
            rate = exp_r / c.harvest_time[obs["difficulty"]]
            a = Action.HARVEST if rate >= self.min_rate else Action.SKIP
        a = self.fallback(a, valid)
        return self.battery_guard(obs, a, valid) if self.guard else a
