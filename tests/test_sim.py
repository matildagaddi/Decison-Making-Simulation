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
