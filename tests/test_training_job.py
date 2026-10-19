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
