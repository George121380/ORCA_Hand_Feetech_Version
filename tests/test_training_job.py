"""A failed long job must retain its status and never proceed to evaluation."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("training_job", Path(__file__).parents[1] / "scripts/run_training_job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def test_selection_and_test_seeds_cannot_overlap(monkeypatch, tmp_path):
    monkeypatch.setattr(job.platform, "system", lambda: "Linux")
    output = tmp_path / "job"
    monkeypatch.setattr(job.sys, "argv", ["job", "--output", str(output),
                       "--selection-seed", "20050", "--test-seed", "20000"])
    with pytest.raises(SystemExit):
        job.main()
    assert not output.exists()


def test_training_failure_stops_job_and_preserves_command(monkeypatch, tmp_path):
    monkeypatch.setattr(job.platform, "system", lambda: "Linux")
    output = tmp_path / "job"
    monkeypatch.setattr(job.sys, "argv", ["job", "--output", str(output)])
    calls = []

    class FailedProcess:
        pid = 123

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 7

    monkeypatch.setattr(job.subprocess, "Popen", FailedProcess)
    with pytest.raises(RuntimeError, match="train exited 7"):
        job.main()
    state = json.loads((output / "job.json").read_text())
    assert len(calls) == 1
    assert state["status"] == "failed"
    assert state["commands"][0]["returncode"] == 7
    assert not state["acceptance_met"]


def test_recovery_keeps_remaining_budget_and_reduces_workers(monkeypatch, tmp_path):
    monkeypatch.setattr(job.platform, "system", lambda: "Linux")
    output = tmp_path / "job"
    (output / "model").mkdir(parents=True)
    arguments = dict(output=str(output), object="pen", angle=15, turns=1, horizon=240,
                     randomization=.3, steps=3000000, num_envs=16, device="cuda", seed=0,
                     resume=None, device_config=None, mass_kg=None, friction=1.,
                     selection_episodes=40, selection_seed=30000, test_episodes=100,
                     test_seed=20000, video=False, reset_log_std=-1., learning_rate=.0001,
                     entropy_coef=.001, target_kl=.01)
    (output / "job.json").write_text(json.dumps(dict(status="failed", arguments=arguments,
                                                    pid=99999999, commands=[], acceptance_met=False)))
    source = output / "model/checkpoint_1500000_steps.zip"
    monkeypatch.setattr(job, "latest_checkpoint", lambda *args: (source, 1500000))
    calls = []

    class FailedProcess:
        pid = 99999998

        def __init__(self, command, **kwargs):
            calls.append(command)

        def wait(self):
            return 7

    monkeypatch.setattr(job.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(job.sys, "argv", ["job", "--output", str(output), "--continue-existing",
                       "--recovery-num-envs", "8"])
    with pytest.raises(RuntimeError, match="train-recovery-1 exited 7"):
        job.main()
    command = calls[0]
    assert command[command.index("--steps") + 1] == "1500000"
    assert command[command.index("--resume") + 1] == str(source)
    assert command[command.index("--num-envs") + 1] == "8"
    assert command[command.index("--learning-rate") + 1] == "0.0001"
    assert "--reset-log-std" not in command
    state = json.loads((output / "job.json").read_text())
    assert state["target_total_steps"] == 3000000
    assert state["arguments"]["steps"] == 3000000
    assert (output / "current-training.log").readlink() == Path("train-recovery-1.log")


@pytest.mark.parametrize("initial_passes", [False, True])
def test_curriculum_stops_at_first_failed_gate(monkeypatch, tmp_path, initial_passes):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1] / "scripts"))
    spec = importlib.util.spec_from_file_location("pen_course", Path(__file__).parents[1] / "scripts/run_pen_curriculum.py")
    course = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(course)
    monkeypatch.setattr(course.platform, "system", lambda: "Linux")
    initial = tmp_path / "initial"
    initial.mkdir()
    arguments = dict(object="pen", angle=15, turns=1, randomization=.3, selection_seed=30000,
                     test_seed=20000, test_episodes=100, num_envs=16, device="cuda", seed=0)
    (initial / "job.json").write_text(json.dumps(dict(status="complete", acceptance_met=initial_passes,
                                                    arguments=arguments)))
    output = tmp_path / "course"
    monkeypatch.setattr(course.sys, "argv", ["course", "--initial-job", str(initial), "--output", str(output)])
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        child = Path(command[command.index("--output") + 1])
        child.mkdir()
        (child / "job.json").write_text(json.dumps(dict(status="complete", acceptance_met=False, test={})))

    monkeypatch.setattr(course.subprocess, "run", run)
    course.main()
    state = json.loads((output / "curriculum.json").read_text())
    assert state["status"] == "quality_gate_failed"
    assert len(calls) == int(initial_passes)
    if calls:
        command = calls[0]
        assert command[command.index("--test-seed") + 1] == "21000"
        assert command[command.index("--resume") + 1] == str(initial / "selected/policy.zip")
