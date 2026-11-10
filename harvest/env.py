"""Harvest dynamics. The env only simulates; belief tracking lives in belief.py.

The robot works through fruits in a fixed order. For the current fruit it can:

    HARVEST        attempt (or retry) picking the fruit
    SKIP           leave the fruit without harvesting
    CHECK          take a noisy quality reading
    RECHARGE       dock and charge (long duration)
    CHECK_WEATHER  read the weather sensor   (only if cfg.weather and not cfg.weather_observed)
    WAIT           idle for wait_time         (only if cfg.allow_wait)

Leaving a fruit (successful harvest, damage, abandon, or skip) always costs
move_time and move_cost. An action is only valid if its worst-case duration fits in
the remaining session time; the episode ends when no action fits, every fruit has
been visited, or the battery reaches zero.

Weather (opt-in) is a Markov chain over {NORMAL, RAIN, HEAT} that steps every
weather_period time units. It scales harvest success and damage probability. The
weather in effect for an action is the weather when the action starts.

Randomness: each fruit gets its own harvest RNG and sensor RNG derived from
(outcome_seed, fruit index), so a fruit's outcomes do not depend on what the policy
did on earlier fruits, and checking a fruit does not change its harvest outcomes.
The weather path is drawn up front from (outcome_seed) and indexed by elapsed time,
so it is the same for every policy in a scenario.
"""
from enum import IntEnum

import numpy as np

from .config import Config
from .field import generate_field


class Action(IntEnum):
    HARVEST = 0
    SKIP = 1
    CHECK = 2
    RECHARGE = 3
    CHECK_WEATHER = 4
    WAIT = 5


def enabled_actions(cfg: Config):
    acts = [Action.HARVEST, Action.SKIP, Action.CHECK, Action.RECHARGE]
    if cfg.weather and not cfg.weather_observed:
        acts.append(Action.CHECK_WEATHER)
    if cfg.allow_wait:
        acts.append(Action.WAIT)
    return acts


def action_duration(cfg: Config, action, difficulty) -> int:
    """Worst-case duration (a harvest may also include the move to the next fruit)."""
    a = Action(action)
    if a == Action.HARVEST:
        return cfg.harvest_time[difficulty] + cfg.move_time
    return {Action.SKIP: cfg.move_time,
            Action.CHECK: cfg.check_time,
            Action.RECHARGE: cfg.recharge_time,
            Action.CHECK_WEATHER: cfg.weather_check_time,
            Action.WAIT: cfg.wait_time}[a]


def action_battery_cost(cfg: Config, action, difficulty) -> float:
    """Worst-case battery drain (negative for RECHARGE)."""
    a = Action(action)
    if a == Action.HARVEST:
        return cfg.harvest_cost[difficulty] + cfg.move_cost
    return {Action.SKIP: cfg.move_cost,
            Action.CHECK: cfg.sensor_cost,
            Action.RECHARGE: -cfg.recharge_rate,
            Action.CHECK_WEATHER: cfg.weather_check_cost,
            Action.WAIT: 0.0}[a]


def generate_weather_path(cfg: Config, outcome_seed) -> np.ndarray:
    """Weather at every elapsed time 0..session_time (inclusive)."""
    if not cfg.weather:
        return np.zeros(cfg.session_time + 1, dtype=int)
    rng = np.random.default_rng([int(outcome_seed), 0, 2])
    n_periods = cfg.session_time // cfg.weather_period + 1
    P = np.asarray(cfg.weather_P, dtype=float)
    w = [int(rng.choice(3, p=np.asarray(cfg.weather_init, dtype=float)))]
    for _ in range(n_periods - 1):
        w.append(int(rng.choice(3, p=P[w[-1]])))
    return np.repeat(w, cfg.weather_period)[:cfg.session_time + 1]


class HarvestEnv:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.done = True

    # ---------- setup ----------
    def reset(self, field_seed=0, outcome_seed=0, fruits=None, weather_path=None):
        c = self.cfg
        self.fruits = list(fruits) if fruits is not None else generate_field(c, field_seed)
        self.outcome_seed = int(outcome_seed)
        self.weather_path = (np.asarray(weather_path, dtype=int) if weather_path is not None
                             else generate_weather_path(c, outcome_seed))
        self._wrng = np.random.default_rng([self.outcome_seed, 0, 3])
        self.weather_readings = []          # (elapsed time, reading) from CHECK_WEATHER
        self.idx = 0
        self.battery = 1.0
        self.time_left = c.session_time
        self.t = 0
        self.done = False
        self.end_reason = None
        self.stats = dict(good=0, bad=0, damaged=0, abandoned=0, skipped=0,
                          checks=0, recharges=0, dead=0, weather_checks=0, waits=0)
        if not self.fruits:
            self._finish("fruits")
        else:
            self._start_fruit()
            if not self.valid_actions():
                self._finish("time")
        return self.observe()

    def _start_fruit(self):
        self.attempts = 0
        self.fail_weather = [0, 0, 0]       # failed attempts on this fruit, by weather
        self.readings = []
        self._hrng = np.random.default_rng([self.outcome_seed, self.idx, 0])
        self._srng = np.random.default_rng([self.outcome_seed, self.idx, 1])

    def _finish(self, reason):
        self.done = True
        self.end_reason = reason

    @property
    def fruit(self):
        return self.fruits[self.idx] if self.idx < len(self.fruits) else None

    @property
    def elapsed(self):
        return self.cfg.session_time - self.time_left

    @property
    def weather_now(self):
        return int(self.weather_path[min(self.elapsed, len(self.weather_path) - 1)])

    # ---------- what the agent sees ----------
    def observe(self):
        c, f = self.cfg, self.fruit
        obs = dict(
            battery=self.battery,
            time_left=self.time_left,
            fruit_index=self.idx,
            fruits_left=len(self.fruits) - self.idx,
            difficulty=f.difficulty if f else None,
            attempts=self.attempts if f else 0,
            readings=tuple(self.readings) if f else (),   # True = sensor read "good"
        )
        if c.weather:
            obs["elapsed"] = self.elapsed
            if c.weather_observed:
                obs["weather"] = self.weather_now
                obs["fail_weather"] = tuple(self.fail_weather) if f else (0, 0, 0)
            else:
                obs["weather"] = None
                obs["weather_readings"] = tuple(self.weather_readings)
        return obs

    def valid_actions(self):
        if self.done or self.fruit is None:
            return []
        d = self.fruit.difficulty
        return [a for a in enabled_actions(self.cfg)
                if action_duration(self.cfg, a, d) <= self.time_left]

    # ---------- dynamics ----------
    def _read_sensor(self, f):
        c = self.cfg
        if self.readings and self._srng.random() < c.sens_rho:
            return self.readings[0]                       # correlated repeat
        return bool(self._srng.random() < (c.sens_tp if f.good else c.sens_fp))

    def _read_weather(self, w):
        u, pick = self._wrng.random(2)
        if u < self.cfg.weather_sens_acc:
            return w
        others = [x for x in range(3) if x != w]
        return others[int(pick * 2)]

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        action = Action(action)
        if action not in self.valid_actions():
            raise ValueError(f"{action.name} is not valid (time_left={self.time_left})")
        c, f = self.cfg, self.fruit
        w = self.weather_now
        m_success, m_damage = c.weather_mult(w)
        info = dict(t=self.t, fruit=self.idx, action=action.name, difficulty=f.difficulty,
                    weather=w if c.weather else None,
                    battery_before=self.battery, time_before=self.time_left,
                    reading=None, success=None, harvested_good=None,
                    damaged=False, abandoned=False, moved=False)
        r, dur, leave = 0.0, 0, False

        if action == Action.HARVEST:
            dur += c.harvest_time[f.difficulty]
            self.battery -= c.harvest_cost[f.difficulty]
            if self.battery > 0:
                u_success, u_damage = self._hrng.random(2)
                if u_success < m_success * f.true_p:
                    info["success"], info["harvested_good"] = True, f.good
                    r += c.r_good if f.good else c.r_bad
                    self.stats["good" if f.good else "bad"] += 1
                    leave = True
                else:
                    info["success"] = False
                    self.attempts += 1
                    self.fail_weather[w] += 1
                    p_dmg = min(1.0, c.damage_base[f.difficulty] * m_damage * self.attempts)
                    if u_damage < p_dmg:
                        info["damaged"] = True
                        r += c.r_damage
                        self.stats["damaged"] += 1
                        leave = True
                    elif c.max_attempts is not None and self.attempts >= c.max_attempts:
                        info["abandoned"] = True
                        self.stats["abandoned"] += 1
                        leave = True

        elif action == Action.SKIP:
            self.stats["skipped"] += 1
            leave = True

        elif action == Action.CHECK:
            dur += c.check_time
            self.battery -= c.sensor_cost
            self.stats["checks"] += 1
            if self.battery > 0:
                reading = self._read_sensor(f)
                self.readings.append(reading)
                info["reading"] = reading

        elif action == Action.RECHARGE:
            dur += c.recharge_time
            self.battery = min(1.0, self.battery + c.recharge_rate)
            self.stats["recharges"] += 1

        elif action == Action.CHECK_WEATHER:
            dur += c.weather_check_time
            self.battery -= c.weather_check_cost
            self.stats["weather_checks"] += 1
            if self.battery > 0:
                reading = self._read_weather(w)
                self.weather_readings.append((self.elapsed, reading))
                info["reading"] = reading

        elif action == Action.WAIT:
            dur += c.wait_time
            self.stats["waits"] += 1

        if leave and self.battery > 0:
            dur += c.move_time
            self.battery -= c.move_cost
            info["moved"] = True

        self.time_left -= dur

        if self.battery <= 0:
            self.battery = 0.0
            r += c.r_dead
            self.stats["dead"] = 1
            self._finish("battery")
        elif self.battery < c.low_battery:
            r += c.r_low * dur

        if leave and not self.done:
            self.idx += 1
            if self.idx >= len(self.fruits):
                self._finish("fruits")
            else:
                self._start_fruit()
        if not self.done and not self.valid_actions():
            self._finish("time")

        self.t += 1
        info.update(duration=dur, reward=r, battery_after=self.battery,
                    time_after=self.time_left, done=self.done, end_reason=self.end_reason)
        return self.observe(), r, self.done, info
