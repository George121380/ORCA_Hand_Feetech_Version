import importlib.util
import json
import os
from pathlib import Path


spec = importlib.util.spec_from_file_location("heartbeat", Path(__file__).parents[1] / "scripts/training_heartbeat.py")
heartbeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heartbeat)


def test_heartbeat_distinguishes_finished_quality_failure_from_live_training(tmp_path):
    state = tmp_path / "supervised"
    job = tmp_path / "job"
    state.mkdir()
    (job / "model").mkdir(parents=True)
    monitor = dict(status="quality_gate_failed", active_job=str(job), stage_name="45-r1", stage_results=[])
    (state / "watchdog.json").write_text(json.dumps(monitor))
    (job / "job.json").write_text(json.dumps(dict(status="complete", phase="complete")))
    report = heartbeat.snapshot(state)
    assert report["needs_attention"] and report["attention_reasons"] == ["quality_gate_failed"]
    monitor["status"] = "monitoring"
    (state / "watchdog.json").write_text(json.dumps(monitor))
    (job / "job.json").write_text(json.dumps(dict(status="running", phase="train")))
    (job / "model/progress.json").write_text(json.dumps(dict(total_steps=100)))
    report = heartbeat.snapshot(state)
    assert not report["needs_attention"] and report["progress"]["total_steps"] == 100
    os.utime(state / "watchdog.json", (1, 1))
    os.utime(job / "model/progress.json", (1, 1))
    report = heartbeat.snapshot(state, now=500)
    assert report["attention_reasons"] == ["stale_watchdog", "stale_training"]


def test_heartbeat_keeps_reporting_unreadable_state(tmp_path):
    report = heartbeat.snapshot(tmp_path / "missing")
    assert report["needs_attention"] and report["attention_reasons"] == ["unreadable_training_state"]
    output = tmp_path / "output"
    heartbeat.write_report(output, report)
    assert json.loads((output / "latest.json").read_text()) == report
    assert len((output / "heartbeats.jsonl").read_text().splitlines()) == 1


def test_heartbeat_reports_checkpoint_selection_progress_without_training(tmp_path):
    state = tmp_path/"selection"
    job = state/"job"
    (job/"model").mkdir(parents=True)
    (state/"watchdog.json").write_text(json.dumps(dict(status="monitoring", active_job=str(job),
        stage_name="35-checkpoint-selection", stage_results=[])))
    (job/"job.json").write_text(json.dumps(dict(status="running", phase="screen")))
    (job/"model/progress.json").write_text(json.dumps(dict(total_steps=0,
        evaluations_completed=7, evaluations_total=49, candidates_total=42)))
    report = heartbeat.snapshot(state)
    assert not report["needs_attention"] and report["phase"] == "screen"
    assert report["progress"]["evaluations_completed"] == 7
