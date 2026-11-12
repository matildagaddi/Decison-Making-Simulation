import json

import numpy as np
import pytest

from harvest.belief import BeliefTracker
from harvest.config import EASY, HARD, Config
from harvest.env import Action, HarvestEnv
from harvest.evaluate import evaluate, make_scenarios, run_episode, summarize
from harvest.field import Fruit, generate_field
from harvest.policies import ThresholdPolicy, baseline_policies


def test_config_json_roundtrip(tmp_path):
    cfg = Config(recharge_time=7, max_attempts=3)
    p = tmp_path / "c.json"
    cfg.save(p)
    assert Config.load(p) == cfg
    json.loads(p.read_text())


def test_same_scenario_is_deterministic():
    cfg = Config()
    sc = make_scenarios(1, seed=3)[0]
    a = run_episode(HarvestEnv(cfg), ThresholdPolicy(cfg), sc)
    b = run_episode(HarvestEnv(cfg), ThresholdPolicy(cfg), sc)
    assert a == b


def test_field_independent_of_policy():
    cfg = Config()
    sc = make_scenarios(1, seed=1)[0]
    fields = []
    for factory in baseline_policies().values():
        env = HarvestEnv(cfg)
        run_episode(env, factory(cfg), sc)
        fields.append(env.fruits)
    assert all(f == fields[0] for f in fields)


def test_checking_does_not_change_harvest_outcomes():
    cfg = Config()
    fruits = [Fruit(0, HARD, True, 0.5)]
    outcomes = []
    for pre_checks in (0, 3):
        env = HarvestEnv(cfg)
        env.reset(outcome_seed=11, fruits=fruits)
        for _ in range(pre_checks):
            env.step(Action.CHECK)
        _, _, _, info = env.step(Action.HARVEST)
        outcomes.append((info["success"], info["damaged"]))
    assert outcomes[0] == outcomes[1]


def test_harvest_success_pays_move_cost():
    cfg = Config()
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, EASY, True, 1.0), Fruit(0, EASY, True, 1.0)])
    _, r, _, info = env.step(Action.HARVEST)
    assert r == cfg.r_good and info["moved"]
    assert info["duration"] == cfg.harvest_time[EASY] + cfg.move_time
    assert env.battery == pytest.approx(1 - cfg.harvest_cost[EASY] - cfg.move_cost)


def test_invalid_action_rejected_and_time_accounting():
    cfg = Config(session_time=5)
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, HARD, True, 0.0)] * 3)
    assert Action.RECHARGE not in env.valid_actions()
    with pytest.raises(ValueError):
        env.step(Action.RECHARGE)
    used = 0
    while not env.done:
        _, _, _, info = env.step(Action.CHECK)
        used += info["duration"]
    assert used == cfg.session_time and env.end_reason == "time"


def test_damage_probability_capped():
    cfg = Config(damage_base={EASY: 0.6, HARD: 0.6})
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, EASY, True, 0.0)])
    damaged = False
    for _ in range(2):          # p = 0.6 then min(1, 1.2) = 1
        _, _, _, info = env.step(Action.HARVEST)
        damaged = info["damaged"]
        if damaged:
            break
    assert damaged


def test_max_attempts_abandons():
    cfg = Config(max_attempts=2, damage_base={EASY: 0.0, HARD: 0.0})
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, EASY, True, 0.0), Fruit(0, EASY, True, 0.0)])
    env.step(Action.HARVEST)
    _, _, _, info = env.step(Action.HARVEST)
    assert info["abandoned"] and env.idx == 1


def test_low_battery_penalty_per_time_unit():
    cfg = Config()
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, EASY, True, 0.0)])
    env.battery = 0.05
    _, r, _, info = env.step(Action.CHECK)
    assert r == pytest.approx(cfg.r_low * cfg.check_time)


def test_belief_independent_and_fully_correlated():
    cfg = Config()
    obs = dict(difficulty=EASY, attempts=0, readings=(True, True))
    indep = BeliefTracker(cfg)(obs)["p_good"]
    pg, tp, fp = cfg.p_good, cfg.sens_tp, cfg.sens_fp
    assert indep == pytest.approx(pg * tp**2 / (pg * tp**2 + (1 - pg) * fp**2))
    corr = BeliefTracker(cfg, sens_rho=1.0)
    one = corr(dict(obs, readings=(True,)))["p_good"]
    assert corr(obs)["p_good"] == pytest.approx(one)   # a copied reading carries no information


def test_correlated_sensor_repeats():
    cfg = Config(sens_rho=1.0)
    env = HarvestEnv(cfg)
    env.reset(outcome_seed=5, fruits=[Fruit(0, EASY, True, 0.5)])
    reads = [env.step(Action.CHECK)[3]["reading"] for _ in range(5)]
    assert len(set(reads)) == 1


def test_policies_use_env_config():
    cfg = Config(recharge_time=50)
    pol = ThresholdPolicy(cfg)
    obs = dict(battery=0.1, time_left=40, difficulty=EASY, attempts=0, readings=())
    assert pol.act(obs, list(Action)) != Action.RECHARGE   # 40 < 50 + 2


def test_evaluate_runs_and_paired_diff_zero_for_reference():
    cfg = Config()
    rows, traj = evaluate(baseline_policies(), cfg, make_scenarios(20), n_log=1)
    s = summarize(rows)
    assert s[0]["diff_vs_ref"] == 0
    assert traj and {"reading", "duration", "reward"} <= set(traj[0])


# ---------------- weather and WAIT ----------------
from harvest.belief import posterior_mean_theta
from harvest.config import HEAT, NORMAL, RAIN
from harvest.env import generate_weather_path


def test_defaults_have_no_weather_or_wait():
    env = HarvestEnv(Config())
    obs = env.reset(fruits=[Fruit(0, EASY, True, 0.5)])
    assert "weather" not in obs
    assert set(env.valid_actions()) == {Action.HARVEST, Action.SKIP, Action.CHECK, Action.RECHARGE}


def test_weather_action_sets():
    fruits = [Fruit(0, EASY, True, 0.5)]
    obs_env = HarvestEnv(Config(weather=True))
    obs = obs_env.reset(fruits=fruits)
    assert obs["weather"] is not None and Action.CHECK_WEATHER not in obs_env.valid_actions()
    hid_env = HarvestEnv(Config(weather=True, weather_observed=False, allow_wait=True))
    obs = hid_env.reset(fruits=fruits)
    assert obs["weather"] is None
    assert {Action.CHECK_WEATHER, Action.WAIT} <= set(hid_env.valid_actions())


def test_weather_path_exogenous_and_deterministic():
    cfg = Config(weather=True)
    a, b = generate_weather_path(cfg, 7), generate_weather_path(cfg, 7)
    assert (a == b).all() and len(a) == cfg.session_time + 1
    assert (a[:cfg.weather_period] == a[0]).all()          # constant within a period
    sc = make_scenarios(1, seed=2)[0]
    paths = []
    for factory in baseline_policies(cfg).values():
        env = HarvestEnv(cfg)
        run_episode(env, factory(cfg), sc)
        paths.append(env.weather_path)
    assert all((p == paths[0]).all() for p in paths)


def test_rain_scales_success_and_counts_failures_by_weather():
    cfg = Config(weather=True, weather_success_mult={NORMAL: 1.0, RAIN: 0.0, HEAT: 1.0},
                 damage_base={EASY: 0.0, HARD: 0.0})
    env = HarvestEnv(cfg)
    rain = np.full(cfg.session_time + 1, RAIN)
    env.reset(fruits=[Fruit(0, EASY, True, 1.0)] * 2, weather_path=rain)
    obs, _, _, info = env.step(Action.HARVEST)
    assert info["success"] is False and info["weather"] == RAIN
    assert obs["fail_weather"] == (0, 1, 0)


def test_rain_raises_damage():
    cfg = Config(weather=True, damage_base={EASY: 0.5, HARD: 0.5},
                 weather_damage_mult={NORMAL: 1.0, RAIN: 2.0, HEAT: 1.0})
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, EASY, True, 0.0)] * 2,
              weather_path=np.full(cfg.session_time + 1, RAIN))
    assert env.step(Action.HARVEST)[3]["damaged"]          # p = min(1, 0.5 * 2 * 1) = 1


def test_posterior_mean_theta():
    assert posterior_mean_theta(3, 5, [(1.0, 4)]) == pytest.approx(3 / 12)
    # quadrature path agrees with the closed form as m -> 1
    assert posterior_mean_theta(3, 5, [(1 - 1e-9, 4)]) == pytest.approx(3 / 12, abs=1e-4)
    # failures under rain are weaker evidence against theta than failures in normal weather
    assert posterior_mean_theta(3, 5, [(0.6, 4)]) > posterior_mean_theta(3, 5, [(1.0, 4)])
    # Monte Carlo check of the weighted posterior
    rng = np.random.default_rng(0)
    th = rng.beta(3, 5, 400_000)
    w = (1 - 0.6 * th) ** 3 * (1 - th) ** 2
    mc = (w * th).sum() / w.sum()
    assert posterior_mean_theta(3, 5, [(0.6, 3), (1.0, 2)]) == pytest.approx(mc, abs=2e-3)


def test_weather_filter():
    cfg = Config(weather=True, weather_observed=False)
    bt = BeliefTracker(cfg)
    P = np.asarray(cfg.weather_P)
    base = dict(difficulty=EASY, attempts=0, readings=())
    assert bt.p_weather(dict(base, elapsed=0, weather=None, weather_readings=())) \
        == pytest.approx(cfg.weather_init)
    seen_rain = dict(base, weather=None, weather_readings=((13, RAIN),))
    assert bt.p_weather(dict(seen_rain, elapsed=15)) == pytest.approx([0, 1, 0])
    assert bt.p_weather(dict(seen_rain, elapsed=22)) == pytest.approx(P[RAIN])
    # a reading that is impossible under the model falls back to the reading itself
    impossible = dict(base, weather=None, weather_readings=((3, RAIN),), elapsed=3)
    assert bt.p_weather(impossible) == pytest.approx([0, 1, 0])
    noisy = BeliefTracker(Config(weather=True, weather_observed=False, weather_sens_acc=0.8))
    pw = noisy.p_weather(dict(seen_rain, elapsed=15))
    assert pw[RAIN] < 1 and pw.sum() == pytest.approx(1)


def test_wait_costs_time_not_battery():
    cfg = Config(allow_wait=True)
    env = HarvestEnv(cfg)
    env.reset(fruits=[Fruit(0, EASY, True, 0.5)])
    env.battery = 0.05
    _, r, _, info = env.step(Action.WAIT)
    assert env.battery == 0.05 and info["duration"] == cfg.wait_time
    assert r == pytest.approx(cfg.r_low * cfg.wait_time)


def test_weather_config_roundtrip(tmp_path):
    cfg = Config(weather=True, weather_observed=False, allow_wait=True, weather_sens_acc=0.9)
    p = tmp_path / "c.json"
    cfg.save(p)
    assert Config.load(p) == cfg


@pytest.mark.parametrize("observed", [True, False])
def test_evaluate_with_weather(observed):
    cfg = Config(weather=True, weather_observed=observed, allow_wait=True)
    pols = baseline_policies(cfg)
    assert "rate (sensor, wait out weather)" in pols
    rows, _ = evaluate(pols, cfg, make_scenarios(10))
    by = {r["policy"]: r for r in rows}
    assert by["rate (sensor, wait out weather)"]["waits"] >= 0
    if not observed:
        assert sum(r["weather_checks"] for r in rows if "check weather" in r["policy"]) > 0
