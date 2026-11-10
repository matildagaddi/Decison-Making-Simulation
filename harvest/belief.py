"""Agent-side beliefs, computed from raw observations.

The tracker's model parameters default to the simulator's, but each can be overridden
to study a misspecified agent (wrong prior, wrong sensor model, ignoring correlation).
"""
import numpy as np

from .config import Config

# midpoint grid on (0, 1) for posteriors that are not Beta (weather-scaled failures)
_GRID = (np.arange(4000) + 0.5) / 4000
_LOG_GRID, _LOG_1M_GRID = np.log(_GRID), np.log1p(-_GRID)


def posterior_mean_theta(a, b, failures):
    """E[theta | failures] for theta ~ Beta(a, b) and failures = [(m, k), ...], where each
    of the k failures had success probability m * theta. The posterior is
    Beta(a, b) * prod (1 - m theta)^k. Exact (Beta) when every m == 1, quadrature otherwise."""
    k_unit = sum(k for m, k in failures if m == 1.0)
    rest = [(m, k) for m, k in failures if m != 1.0 and k > 0]
    b = b + k_unit
    if not rest:
        return a / (a + b)
    logw = (a - 1) * _LOG_GRID + (b - 1) * _LOG_1M_GRID
    for m, k in rest:
        logw = logw + k * np.log1p(-m * _GRID)
    w = np.exp(logw - logw.max())
    return float((w * _GRID).sum() / w.sum())


class BeliefTracker:
    def __init__(self, cfg: Config, prior_beta=None, p_good=None,
                 sens_tp=None, sens_fp=None, sens_rho=None):
        self.prior_beta = prior_beta if prior_beta is not None else cfg.prior_beta
        self.cfg = cfg
        self.p_good_prior = p_good if p_good is not None else cfg.p_good
        self.sens_tp = sens_tp if sens_tp is not None else cfg.sens_tp
        self.sens_fp = sens_fp if sens_fp is not None else cfg.sens_fp
        self.sens_rho = sens_rho if sens_rho is not None else cfg.sens_rho

    # ---------- weather ----------
    def p_weather(self, obs):
        """Belief over current weather (NORMAL, RAIN, HEAT). None when weather is off.

        Hidden weather: exact HMM filter over weather periods using CHECK_WEATHER readings
        only. Harvest outcomes also carry weak evidence about the weather; it is ignored here.
        """
        c = self.cfg
        if not c.weather:
            return None
        if obs.get("weather") is not None:
            pw = np.zeros(3)
            pw[obs["weather"]] = 1.0
            return pw
        P = np.asarray(c.weather_P, dtype=float)
        acc = c.weather_sens_acc
        pw = np.asarray(c.weather_init, dtype=float)
        period = 0
        for e, reading in obs.get("weather_readings", ()):
            pw = pw @ np.linalg.matrix_power(P, e // c.weather_period - period)
            period = e // c.weather_period
            like = np.full(3, (1 - acc) / 2)
            like[reading] = acc
            post = pw * like
            # evidence impossible under the model (e.g. a misspecified agent): trust the reading
            pw = post / post.sum() if post.sum() > 0 else like / like.sum()
        return pw @ np.linalg.matrix_power(P, obs["elapsed"] // c.weather_period - period)

    def p_success(self, obs, pw=None):
        """Predictive P(next attempt succeeds).

        No weather: posterior mean of Beta(a, b + failures), i.e. a / (a + b + k).
        Observed weather: failures are counted per weather type; P = m_w * E[theta | counts].
        Hidden weather: approximation, every failure and the next attempt use the belief-mean
        multiplier m_bar = sum_w P(w) m_w.
        """
        a, b = self.prior_beta[obs["difficulty"]]
        c = self.cfg
        if not c.weather:
            return a / (a + b + obs["attempts"])
        mult = c.weather_success_mult
        if obs.get("weather") is not None:
            fails = [(mult[w], k) for w, k in enumerate(obs["fail_weather"])]
            return mult[obs["weather"]] * posterior_mean_theta(a, b, fails)
        pw = self.p_weather(obs) if pw is None else pw
        m_bar = float(sum(pw[w] * mult[w] for w in range(3)))
        return m_bar * posterior_mean_theta(a, b, [(m_bar, obs["attempts"])])

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
        pw = self.p_weather(obs)
        return dict(p_success=self.p_success(obs, pw), p_good=self.p_good(obs), p_weather=pw)
