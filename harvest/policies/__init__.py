from .base import Policy
from .baselines import GreedyPolicy, RandomPolicy, RatePolicy, ThresholdPolicy


def baseline_policies():
    """name -> factory(cfg). Factories are called once per episode with the env's Config."""
    return {
        "random": lambda cfg: RandomPolicy(cfg),
        "greedy": lambda cfg: GreedyPolicy(cfg),
        "threshold (no sensor)": lambda cfg: ThresholdPolicy(cfg, use_sensor=False),
        "threshold (sensor)": lambda cfg: ThresholdPolicy(cfg, use_sensor=True),
        "threshold (sensor, strict)": lambda cfg: ThresholdPolicy(
            cfg, recharge_below=0.3, min_p_good=0.7, min_p_success=0.4),
        "rate (sensor)": lambda cfg: RatePolicy(cfg),
        "rate (sensor, picky)": lambda cfg: RatePolicy(cfg, min_rate=0.8),
    }


__all__ = ["Policy", "RandomPolicy", "GreedyPolicy", "ThresholdPolicy", "RatePolicy",
           "baseline_policies"]
