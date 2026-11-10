from ..belief import BeliefTracker
from ..config import Config
from ..env import Action, action_battery_cost


class Policy:
    """A policy sees the observation and the list of valid actions and returns an Action.

    Policies always use the Config they are constructed with; build them from the same
    Config as the env (evaluate.py does this).
    """
    name = "policy"

    def __init__(self, cfg: Config, belief: BeliefTracker = None):
        self.cfg = cfg
        self.belief = belief or BeliefTracker(cfg)
        self.rng = None

    def reset(self, rng):
        self.rng = rng

    def act(self, obs, valid):
        raise NotImplementedError

    # ---- helpers ----
    def would_die(self, obs, action):
        return obs["battery"] - action_battery_cost(self.cfg, action, obs["difficulty"]) <= 0

    def battery_guard(self, obs, action, valid):
        """Avoid an action that could drain the battery: recharge if possible, else pick
        a valid action that is survivable (cheapest first). If none, keep the choice."""
        if not self.would_die(obs, action):
            return action
        if Action.RECHARGE in valid:
            return Action.RECHARGE
        safe = [a for a in (Action.SKIP, Action.CHECK, Action.HARVEST)
                if a in valid and not self.would_die(obs, a)]
        return safe[0] if safe else action

    @staticmethod
    def fallback(action, valid):
        """Replace an action that does not fit in the remaining time."""
        if action in valid:
            return action
        return Action.SKIP if Action.SKIP in valid else valid[0]
