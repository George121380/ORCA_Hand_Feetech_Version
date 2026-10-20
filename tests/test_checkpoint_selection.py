import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def selector(monkeypatch):
    scripts = Path(__file__).parents[1]/"scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("select_checkpoints", scripts/"select_checkpoints.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocol(selector, tmp_path):
    # Orchestration fixtures, never loaded as a policy; runtime separately
    # verifies real checkpoint contracts/weights before any scoring.
    candidates = []
    for i in range(2):
        root = tmp_path/f"source{i}"
        root.mkdir()
        path = root/"policy.zip"
        path.write_bytes(f"test-fixture-{i}".encode())
        (root/"config.json").write_text(json.dumps(dict(trained=True)))
        digest = selector.sha256(path)
        candidates.append(dict(id=digest, checkpoint=str(path), sha256=digest, steps=100+i,
                               config_sha256=selector.sha256(root/"config.json")))
    return dict(candidates=candidates, workers=2, top_k=2,
                screen=dict(seed=140000, episodes=40),
                confirmation=dict(seed=820000, episodes=200),
                test=dict(seed=830000, episodes=100), video=True)


def metrics(candidate, episodes, seed, score, baseline):
    return dict(policy=baseline or candidate["checkpoint"], episodes=episodes,
                success_rate=score, drop_rate=0., mean_final_error_deg=10.,
                acceptance_met=episodes >= 100 and score >= .8,
                trials=[dict(seed=seed+i) for i in range(episodes)])


def test_fresh_confirmation_selects_winner_before_one_formal_block_and_caches_resume(selector, tmp_path):
    p = protocol(selector, tmp_path)
    calls = []
    first = p["candidates"][0]["id"]

    def evaluate(candidate, output, episodes, seed, baseline, video):
        calls.append((candidate["id"], episodes, seed, baseline, video))
        if seed == 140000:
            score = .95 if candidate["id"] == first else .9
        elif seed == 820000:
            score = .81 if candidate["id"] == first else .9
        else:
            assert candidate["id"] != first
            assert (output.parent/"selected/policy.zip").exists()
            score = .82 if baseline is None else 0.
        return metrics(candidate, episodes, seed, score, baseline)

    state = tmp_path/"state"
    job = selector.SelectionJob(p, state, evaluate).run()
    assert job["acceptance_met"] and job["test"]["episodes"] == 100
    assert job["selected_sha256"] == p["candidates"][1]["sha256"]
    assert job["validation"]["is_formal_evaluation"] is False
    assert len(calls) == 8
    assert sum(n == 100 and seed == 830000 and b is None for _, n, seed, b, _ in calls) == 1
    # Emulate recovery after all artifacts exist but before completion persists.
    job["status"] = "running"
    (state/"job/job.json").write_text(json.dumps(job))
    def forbidden(*args):
        raise AssertionError("Matching cached evaluations must not be repeated")
    resumed = selector.SelectionJob(p, state, forbidden).run()
    assert resumed["acceptance_met"] and resumed["selected_sha256"] == job["selected_sha256"]


def test_failed_confirmation_never_consumes_formal_test(selector, tmp_path):
    p = protocol(selector, tmp_path)
    calls = []
    def evaluate(candidate, output, episodes, seed, baseline, video):
        calls.append(seed)
        assert seed != 830000
        return metrics(candidate, episodes, seed, .7, baseline)
    state = tmp_path/"state"
    job = selector.SelectionJob(p, state, evaluate).run()
    assert len(calls) == 4 and not job["acceptance_met"] and "test" not in job
    assert "formal test not consumed" in job["error"]
    assert json.loads((state/"watchdog.json").read_text())["status"] == "quality_gate_failed"
    p["test"]["seed"] += 1000
    with pytest.raises(ValueError, match="Frozen selection protocol"):
        selector.SelectionJob(p, state, evaluate).run()


def test_protocol_rejects_overlap_and_excessive_budget_and_cache_requires_exact_seeds(selector, tmp_path):
    p = protocol(selector, tmp_path)
    p["test"]["seed"] = p["confirmation"]["seed"]
    with pytest.raises(ValueError, match="overlap"):
        selector.validate_protocol(p)
    p["test"]["seed"] = 830000
    p["workers"] = 5
    with pytest.raises(ValueError, match="workers"):
        selector.validate_protocol(p)
    c = p["candidates"][0]
    output = tmp_path/"cache"
    output.mkdir()
    (output/"metrics.json").write_text(json.dumps(metrics(c, 40, 140000, .7, None)))
    (output/"checkpoint.sha256").write_text(c["sha256"])
    assert selector.cached_metrics(output, c, 40, 140000)
    assert selector.cached_metrics(output, c, 40, 140001) is None
    (output/"checkpoint.sha256").write_text("wrong")
    assert selector.cached_metrics(output, c, 40, 140000) is None
