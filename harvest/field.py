"""Field generation, seeded independently of the harvest/sensor outcomes."""
from dataclasses import dataclass

import numpy as np

from .config import EASY, HARD, Config


@dataclass(frozen=True)
class Fruit:
    plant: int
    difficulty: int
    good: bool
    true_p: float


def generate_field(cfg: Config, seed) -> list:
    """Fruits in visiting order. The same seed always gives the same field."""
    rng = np.random.default_rng(seed)
    fruits = []
    for plant in range(cfg.n_plants):
        n = max(cfg.min_fruits_per_plant, int(rng.poisson(cfg.mean_fruits_per_plant)))
        for _ in range(n):
            d = HARD if rng.random() < cfg.p_hard else EASY
            good = bool(rng.random() < cfg.p_good)
            a, b = cfg.true_beta[d]
            fruits.append(Fruit(plant, d, good, float(rng.beta(a, b))))
    return fruits
