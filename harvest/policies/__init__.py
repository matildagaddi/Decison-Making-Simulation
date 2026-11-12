from .base import Policy
from .baselines import GreedyPolicy, RandomPolicy, RatePolicy, ThresholdPolicy


def baseline_policies(cfg=None):
    """name -> factory(cfg). Factories are called once per episode with the env's Config.
    Weather-aware variants are added when cfg has weather (or WAIT) enabled."""
    pols = {
        "random": lambda cfg: RandomPolicy(cfg),
        "greedy": lambda cfg: GreedyPolicy(cfg),
        "threshold (no sensor)": lambda cfg: ThresholdPolicy(cfg, use_sensor=False),
        "threshold (sensor)": lambda cfg: ThresholdPolicy(cfg, use_sensor=True),
        "threshold (sensor, strict)": lambda cfg: ThresholdPolicy(
            cfg, recharge_below=0.3, min_p_good=0.7, min_p_success=0.4),
        "rate (sensor)": lambda cfg: RatePolicy(cfg),
        "rate (sensor, picky)": lambda cfg: RatePolicy(cfg, min_rate=0.8),
    }
    if cfg is not None and cfg.weather:
        if not cfg.weather_observed:
            pols["rate (sensor, check weather/10)"] = lambda cfg: RatePolicy(cfg, weather_check_every=10)
        if cfg.allow_wait:
            every = None if cfg.weather_observed else 10
            pols["rate (sensor, wait out weather)"] = lambda cfg: RatePolicy(
                cfg, weather_check_every=every, wait_bad_weather=True)
    return pols


__all__ = ["Policy", "RandomPolicy", "GreedyPolicy", "ThresholdPolicy", "RatePolicy",
           "baseline_policies"]
