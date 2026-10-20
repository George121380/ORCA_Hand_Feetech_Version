"""Independent, read-only training heartbeat; runs even after the watchdog exits.

This writes server-side status and attention records. It does not wake a chat
agent or send messages. Install as a separate systemd user service.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import tempfile
import time


def snapshot(state_dir, now=None):
    now = time.time() if now is None else now
    report = dict(timestamp=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                  state_dir=str(state_dir), needs_attention=False)
    try:
        monitor_path = state_dir / "watchdog.json"
        state = json.loads(monitor_path.read_text())
        job_path = Path(state["active_job"]) / "job.json"
        job = json.loads(job_path.read_text())
        report.update(monitor_status=state["status"], stage=state["stage_name"],
                      active_job=state["active_job"], job_status=job["status"], phase=job.get("phase"),
                      monitor_age_seconds=max(0, now-monitor_path.stat().st_mtime),
                      quality_attempt=state.get("quality_attempt", 0), results=state["stage_results"],
                      latest_test=job.get("test"), target_total_steps=job.get("target_total_steps"))
        progress = job_path.parent / job.get("current_model", "model") / "progress.json"
        if progress.exists():
            data = json.loads(progress.read_text())
            report["progress"] = {k: data.get(k) for k in ("total_steps", "steps_per_second", "elapsed_s")}
            report["progress"].update({k: data[k] for k in (
                "evaluations_completed", "evaluations_total", "candidates_total") if k in data})
            report["progress_age_seconds"] = max(0, now-progress.stat().st_mtime)
        reasons = []
        if state["status"] in ("quality_gate_failed", "recovery_exhausted", "needs_review"):
            reasons.append(state["status"])
        if job["status"] == "running" and report["monitor_age_seconds"] > 120:
            reasons.append("stale_watchdog")
        if (job["status"] == "running" and job.get("phase") == "train"
                and report.get("progress_age_seconds", 0) > 300):
            reasons.append("stale_training")
        report.update(needs_attention=bool(reasons), attention_reasons=reasons)
    except (OSError, ValueError, KeyError, TypeError) as error:
        report.update(needs_attention=True, attention_reasons=["unreadable_training_state"], error=str(error))
    return report


def write_report(output, report):
    output.mkdir(parents=True, exist_ok=True)
    # A service startup and an authorized --once inspection can overlap.
    # Serialize writers and use a unique temporary file so neither can rename
    # or remove the other writer's pending snapshot.
    with (output / '.write.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        with tempfile.NamedTemporaryFile(mode='w',dir=output,prefix='.latest-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name)
            stream.write(json.dumps(report,indent=2)+'\n')
        try:
            temporary.replace(output/'latest.json')
        finally:
            temporary.unlink(missing_ok=True)
        with (output / "heartbeats.jsonl").open("a") as log:
            log.write(json.dumps(report) + "\n")
    print(json.dumps({k: report.get(k) for k in ("timestamp", "monitor_status", "stage",
                                               "job_status", "progress", "needs_attention",
                                               "attention_reasons")}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=300)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    if args.interval < 10:
        parser.error("Heartbeat interval must be at least 10 seconds")
    while True:
        report = snapshot(args.state_dir)
        try:
            process = subprocess.run(["systemctl", "--user", "is-active", "orca-rl-watchdog.service"],
                                     capture_output=True, text=True, timeout=10)
            report["watchdog_service"] = process.stdout.strip()
            if report.get("job_status") == "running" and process.stdout.strip() not in ("active", "activating"):
                report["needs_attention"] = True
                report.setdefault("attention_reasons", []).append("watchdog_service_inactive")
        except (OSError, subprocess.TimeoutExpired) as error:
            report["service_check_error"] = str(error)
        write_report(args.output, report)
        if args.once:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
