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
