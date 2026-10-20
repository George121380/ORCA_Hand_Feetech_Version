import copy
import json
from pathlib import Path

import pytest


@pytest.fixture
def experiment(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    import run_imitation_probe
    return run_imitation_probe


def result(successes, drops=0):
    return dict(trials=[dict(seed=1020000+i,direction=(-1)**i,initial_position=[.1,.2,.3],
                            initial_rotation=[1],goal_rotation=[2],is_success=i<successes,
                            dropped=successes <= i < successes+drops) for i in range(200)])


def test_development_gate_is_integer_paired_and_never_formal_acceptance(experiment):
    source = result(140)
    comparison = experiment.continuation_gate(source,result(150))
    assert comparison["continuation_recommended"] and not comparison["acceptance_met"]
    assert comparison["gained"] == 10 and comparison["lost"] == 0
    assert not experiment.continuation_gate(source,result(149))["continuation_recommended"]
    assert not experiment.continuation_gate(result(141),result(150))["continuation_recommended"]
    assert not experiment.continuation_gate(source,result(150,3))["continuation_recommended"]
    wrong = result(180)
    wrong["trials"][0]["initial_rotation"] = [3]
    with pytest.raises(ValueError,match="paired"):
        experiment.continuation_gate(source,wrong)


def test_pipeline_is_bounded_preserves_completed_run_and_consumes_no_formal_test(experiment,tmp_path,monkeypatch):
    source = tmp_path/"source.zip"
    source.write_bytes(b"unloaded orchestration fixture")
    (tmp_path/"config.json").write_text("{}")
    prior = tmp_path/"prior.json"
    prior.write_text("{}")
    monkeypatch.setattr(experiment,"load_config",lambda path:(None,dict(fingerprint="contract")))
    monkeypatch.setattr(experiment,"checkpoint_steps",lambda *a,**kw:1234)
    job = experiment.Experiment(source,prior,tmp_path/"state")
    calls = []
    def command(name,argv,timeout):
        calls.append((name,argv,timeout))
        assert all("1030000" not in str(a) for a in argv)
        output = Path(argv[argv.index("--output")+1])
        output.mkdir()
        if name == "collect-training128":
            assert timeout == 600 and argv[argv.index("--seed")+1] == "32"
            (output/"probe.json").write_text(json.dumps(dict(status="complete",seed=32,episodes=128,
                checkpoint_sha256=experiment.sha256(source),config_sha256=experiment.sha256(tmp_path/"config.json"))))
        else:
            assert name == "offline-imitation" and timeout == 180
            (output/"policy.zip").write_bytes(b"unloaded imitation fixture")
            (output/"config.json").write_text("{}")
            (output/"imitation.json").write_text(json.dumps(dict(source_sha256=experiment.sha256(source),
                datasets=[dict(sha256=experiment.sha256(p)) for p in
                          (prior,job.output/"training128/probe.json")])))
    def evaluate(name,checkpoint):
        assert name in ("development-source","development-imitation")
        job.job["evaluations_completed"] += 1
        return result(140 if name == "development-source" else 155)
    monkeypatch.setattr(job,"command",command)
    monkeypatch.setattr(job,"evaluate",evaluate)
    report = job.run()
    assert len(calls) == 2 and report["status"] == "complete" and not report["acceptance_met"]
    assert report["comparison"]["continuation_recommended"]
    assert job.protocol["formal_test"] is None and job.protocol["maximum_attempts"] == 3
    progress = json.loads((job.output/"model/progress.json").read_text())
    assert progress["total_steps"] == 0 and progress["evaluations_completed"] == 2
    # Re-entering a completed fixed experiment cannot train or score again.
    monkeypatch.setattr(job,"command",lambda *a:pytest.fail("completed run executed again"))
    assert job.run() == report
    frozen = copy.deepcopy(job.protocol)
    job.protocol["development"]["seed"] += 1
    with pytest.raises(ValueError,match="Frozen"):
        job.run()
    assert json.loads((job.state_dir/"protocol.json").read_text()) == frozen
