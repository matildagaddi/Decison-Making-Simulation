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
    """Harvest only if immediate expected reward per unit harvest time >= min_rate.

    Weather options (only used when cfg.weather is on):
      weather_check_every  hidden weather: CHECK_WEATHER when the last reading is at least
                           this old (None = never check)
      wait_bad_weather     if WAIT is enabled and the fruit would pass the rate test in normal
                           weather but fails it under the current weather belief, WAIT
    """
    name = "rate"

    def __init__(self, cfg, recharge_below=0.25, min_rate=0.5, use_sensor=True, guard=True,
                 weather_check_every=None, wait_bad_weather=False, **kw):
        super().__init__(cfg, **kw)
        self.recharge_below, self.min_rate = recharge_below, min_rate
        self.use_sensor, self.guard = use_sensor, guard
        self.weather_check_every, self.wait_bad_weather = weather_check_every, wait_bad_weather

    def _rate(self, obs, p_success, p_good):
        c = self.cfg
        return p_success * (p_good * c.r_good + (1 - p_good) * c.r_bad) / c.harvest_time[obs["difficulty"]]

    def _needs_weather_check(self, obs, valid):
        if self.weather_check_every is None or Action.CHECK_WEATHER not in valid:
            return False
        reads = obs.get("weather_readings", ())
        return not reads or obs["elapsed"] - reads[-1][0] >= self.weather_check_every

    def act(self, obs, valid):
        c = self.cfg
        if (obs["battery"] < self.recharge_below and Action.RECHARGE in valid
                and obs["time_left"] > c.recharge_time + 2):
            return Action.RECHARGE
        if self._needs_weather_check(obs, valid):
            a = Action.CHECK_WEATHER
        elif self.use_sensor and obs["attempts"] == 0 and not obs["readings"]:
            a = Action.CHECK
        else:
            b = self.belief(obs)
            ok = self._rate(obs, b["p_success"], b["p_good"]) >= self.min_rate
            a = Action.HARVEST if ok else Action.SKIP
            if not ok and self.wait_bad_weather and Action.WAIT in valid and c.weather:
                normal = dict(obs, weather=0, fail_weather=obs.get("fail_weather", (obs["attempts"], 0, 0)))
                if self._rate(obs, self.belief.p_success(normal), b["p_good"]) >= self.min_rate:
                    a = Action.WAIT
        a = self.fallback(a, valid)
        return self.battery_guard(obs, a, valid) if self.guard else a
