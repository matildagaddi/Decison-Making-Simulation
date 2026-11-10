"""Simulation parameters. One Config instance is shared by the env, beliefs and policies."""
import json
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

EASY, HARD = 0, 1


@dataclass
class Config:
    # ---- field generation ----
    n_plants: int = 20
    mean_fruits_per_plant: float = 5.0   # Poisson mean
    min_fruits_per_plant: int = 1        # Poisson draws are clipped up to this
    p_hard: float = 0.4
    p_good: float = 0.6
    # true harvest-success probability ~ Beta(a, b) by difficulty
    true_beta: dict = field(default_factory=lambda: {EASY: (8, 2), HARD: (3, 5)})

    # ---- agent prior over success probability (Beta params) by difficulty ----
    prior_beta: dict = field(default_factory=lambda: {EASY: (4, 1), HARD: (2, 3)})

    # ---- session ----
    session_time: int = 200     # time budget per session (time units)
    gamma: float = 0.99         # discount PER TIME UNIT (an action of duration d discounts by gamma**d)

    # ---- battery (fraction of full charge) ----
    harvest_cost: dict = field(default_factory=lambda: {EASY: 0.02, HARD: 0.06})
    sensor_cost: float = 0.005
    move_cost: float = 0.005    # paid every time the robot leaves a fruit (harvest, damage, skip, abandon)
    recharge_rate: float = 0.30
    low_battery: float = 0.10

    # ---- durations (time units) ----
    harvest_time: dict = field(default_factory=lambda: {EASY: 1, HARD: 2})
    check_time: int = 1
    move_time: int = 1
    recharge_time: int = 10

    # ---- damage on a failed attempt: P = min(1, base * failed_attempts_so_far) ----
    damage_base: dict = field(default_factory=lambda: {EASY: 0.05, HARD: 0.10})
    max_attempts: Optional[int] = None   # abandon the fruit after this many failures (None = no limit)

    # ---- quality sensor ----
    sens_tp: float = 0.85       # P(reads good | good)
    sens_fp: float = 0.15       # P(reads good | bad)
    sens_rho: float = 0.0       # reading correlation: each repeat reading copies the first one w.p. rho

    # ---- rewards ----
    r_good: float = 2.0
    r_bad: float = -1.0
    r_damage: float = -2.0
    r_low: float = -2.0         # PER TIME UNIT spent below low_battery
    r_dead: float = -10.0       # battery hits zero (episode ends)

    # ---- serialisation ----
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        for f in fields(cls):
            v = d.get(f.name)
            if isinstance(v, dict):   # JSON turns int keys into strings and tuples into lists
                d[f.name] = {int(k): tuple(x) if isinstance(x, list) else x for k, x in v.items()}
        return cls(**d)

    def save(self, path):
        with open(path, "w") as fh:
            json.dump(self.to_dict(), fh, indent=2)

    @classmethod
    def load(cls, path):
        with open(path) as fh:
            return cls.from_dict(json.load(fh))
