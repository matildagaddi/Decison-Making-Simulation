"""Harvest dynamics. The env only simulates; belief tracking lives in belief.py.

The robot works through fruits in a fixed order. For the current fruit it can:

    HARVEST   attempt (or retry) picking the fruit
    SKIP      leave the fruit without harvesting
    CHECK     take a noisy quality reading
    RECHARGE  dock and charge (long duration)

Leaving a fruit (successful harvest, damage, abandon, or skip) always costs
move_time and move_cost. An action is only valid if its worst-case duration fits in
the remaining session time; the episode ends when no action fits, every fruit has
been visited, or the battery reaches zero.

Randomness: each fruit gets its own harvest RNG and sensor RNG derived from
(outcome_seed, fruit index), so a fruit's outcomes do not depend on what the policy
did on earlier fruits, and checking a fruit does not change its harvest outcomes.
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


def action_duration(cfg: Config, action, difficulty) -> int:
    """Worst-case duration (a harvest may also include the move to the next fruit)."""
    return {Action.HARVEST: cfg.harvest_time[difficulty] + cfg.move_time,
            Action.SKIP: cfg.move_time,
            Action.CHECK: cfg.check_time,
            Action.RECHARGE: cfg.recharge_time}[Action(action)]


def action_battery_cost(cfg: Config, action, difficulty) -> float:
    """Worst-case battery drain (negative for RECHARGE)."""
    return {Action.HARVEST: cfg.harvest_cost[difficulty] + cfg.move_cost,
            Action.SKIP: cfg.move_cost,
            Action.CHECK: cfg.sensor_cost,
            Action.RECHARGE: -cfg.recharge_rate}[Action(action)]


class HarvestEnv:
    def __init__(self, cfg: Config = None):
        self.cfg = cfg or Config()
        self.done = True

    # ---------- setup ----------
    def reset(self, field_seed=0, outcome_seed=0, fruits=None):
        c = self.cfg
        self.fruits = list(fruits) if fruits is not None else generate_field(c, field_seed)
        self.outcome_seed = int(outcome_seed)
        self.idx = 0
        self.battery = 1.0
        self.time_left = c.session_time
        self.t = 0
        self.done = False
        self.end_reason = None
        self.stats = dict(good=0, bad=0, damaged=0, abandoned=0, skipped=0,
                          checks=0, recharges=0, dead=0)
        if not self.fruits:
            self._finish("fruits")
        else:
            self._start_fruit()
            if not self.valid_actions():
                self._finish("time")
        return self.observe()

    def _start_fruit(self):
        self.attempts = 0
        self.readings = []
        self._hrng = np.random.default_rng([self.outcome_seed, self.idx, 0])
        self._srng = np.random.default_rng([self.outcome_seed, self.idx, 1])

    def _finish(self, reason):
        self.done = True
        self.end_reason = reason

    @property
    def fruit(self):
        return self.fruits[self.idx] if self.idx < len(self.fruits) else None

    # ---------- what the agent sees ----------
    def observe(self):
        f = self.fruit
        return dict(
            battery=self.battery,
            time_left=self.time_left,
            fruit_index=self.idx,
            fruits_left=len(self.fruits) - self.idx,
            difficulty=f.difficulty if f else None,
            attempts=self.attempts if f else 0,
            readings=tuple(self.readings) if f else (),   # True = sensor read "good"
        )

    def valid_actions(self):
        if self.done or self.fruit is None:
            return []
        d = self.fruit.difficulty
        return [a for a in Action if action_duration(self.cfg, a, d) <= self.time_left]

    # ---------- dynamics ----------
    def _read_sensor(self, f):
        c = self.cfg
        if self.readings and self._srng.random() < c.sens_rho:
            return self.readings[0]                       # correlated repeat
        return bool(self._srng.random() < (c.sens_tp if f.good else c.sens_fp))

    def step(self, action):
        if self.done:
            raise RuntimeError("step() called on a finished episode")
        action = Action(action)
        if action not in self.valid_actions():
            raise ValueError(f"{action.name} does not fit in time_left={self.time_left}")
        c, f = self.cfg, self.fruit
        info = dict(t=self.t, fruit=self.idx, action=action.name, difficulty=f.difficulty,
                    battery_before=self.battery, time_before=self.time_left,
                    reading=None, success=None, harvested_good=None,
                    damaged=False, abandoned=False, moved=False)
        r, dur, leave = 0.0, 0, False

        if action == Action.HARVEST:
            dur += c.harvest_time[f.difficulty]
            self.battery -= c.harvest_cost[f.difficulty]
            if self.battery > 0:
                u_success, u_damage = self._hrng.random(2)
                if u_success < f.true_p:
                    info["success"], info["harvested_good"] = True, f.good
                    r += c.r_good if f.good else c.r_bad
                    self.stats["good" if f.good else "bad"] += 1
                    leave = True
                else:
                    info["success"] = False
                    self.attempts += 1
                    if u_damage < min(1.0, c.damage_base[f.difficulty] * self.attempts):
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
