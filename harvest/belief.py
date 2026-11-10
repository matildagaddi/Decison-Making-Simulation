"""Agent-side beliefs, computed from raw observations.

The tracker's model parameters default to the simulator's, but each can be overridden
to study a misspecified agent (wrong prior, wrong sensor model, ignoring correlation).
"""
from .config import Config


class BeliefTracker:
    def __init__(self, cfg: Config, prior_beta=None, p_good=None,
                 sens_tp=None, sens_fp=None, sens_rho=None):
        self.prior_beta = prior_beta if prior_beta is not None else cfg.prior_beta
        self.p_good_prior = p_good if p_good is not None else cfg.p_good
        self.sens_tp = sens_tp if sens_tp is not None else cfg.sens_tp
        self.sens_fp = sens_fp if sens_fp is not None else cfg.sens_fp
        self.sens_rho = sens_rho if sens_rho is not None else cfg.sens_rho

    def p_success(self, obs):
        """Posterior mean of Beta(a, b + failures): every failed attempt is one more 'b'."""
        a, b = self.prior_beta[obs["difficulty"]]
        return a / (a + b + obs["attempts"])

    def _p_read(self, reading, good):
        p = self.sens_tp if good else self.sens_fp
        return p if reading else 1.0 - p

    def likelihood(self, readings, good):
        """P(readings | quality) under the correlated-repeat sensor model."""
        if not readings:
            return 1.0
        first = readings[0]
        like = self._p_read(first, good)
        for r in readings[1:]:
            like *= self.sens_rho * (r == first) + (1 - self.sens_rho) * self._p_read(r, good)
        return like

    def p_good(self, obs):
        pg = self.p_good_prior
        lg = self.likelihood(obs["readings"], True)
        lb = self.likelihood(obs["readings"], False)
        return pg * lg / (pg * lg + (1 - pg) * lb)

    def __call__(self, obs):
        return dict(p_success=self.p_success(obs), p_good=self.p_good(obs))
