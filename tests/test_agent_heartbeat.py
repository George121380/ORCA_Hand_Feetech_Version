import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace


spec = importlib.util.spec_from_file_location('pulse', Path(__file__).parents[1] / 'scripts/queue_rl_heartbeat.py')
heartbeat = importlib.util.module_from_spec(spec)
spec.loader.exec_module(heartbeat)


def test_os_pulse_deduplicates_pending_checks_and_records_actual_ack(monkeypatch, tmp_path):
    path = tmp_path / 'heartbeat.json'
    path.write_text(json.dumps(dict(enabled=True, codex='/bin/codex', cwd=str(tmp_path),
                                    threadId='same-thread', prompt='Inspect training',
                                    queued_sequence=0, acknowledged_sequence=0)))
    calls = []

    def queue(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stdout='Queued message for same-thread', stderr='')

    monkeypatch.setattr(heartbeat.subprocess, 'run', queue)
    first = heartbeat.pulse(path, now=1000)
    assert first['queued_sequence'] == 1 and first['acknowledged_sequence'] == 0
    assert calls[0][1:4] == ['queue', '--thread', 'same-thread']
    second = heartbeat.pulse(path, now=1300)
    assert len(calls) == 1 and second['status'] == 'awaiting_agent_ack'
    stale = heartbeat.pulse(path, now=2000)
    assert stale['needs_attention'] and len(calls) == 1
    ack = heartbeat.pulse(path, ack=True, note='SSH checked: job running', now=2100)
    assert ack['acknowledged_sequence'] == 1 and ack['last_inspection_note'].startswith('SSH checked')
    assert ack['last_inspection_at'] == ack['last_ack_at']
    heartbeat.pulse(path, now=2200)
    assert len(calls) == 2
    heartbeat.pulse(path, disable=True, now=2300)
    heartbeat.pulse(path, now=2400)
    assert len(calls) == 2


def test_queue_failure_is_not_counted_as_delivered(monkeypatch, tmp_path):
    path = tmp_path / 'heartbeat.json'
    path.write_text(json.dumps(dict(enabled=True, codex='/bin/codex', cwd=str(tmp_path),
                                    threadId='same-thread', prompt='Inspect training')))
    monkeypatch.setattr(heartbeat.subprocess, 'run', lambda *a, **k:
                        SimpleNamespace(returncode=1, stdout='', stderr='daemon unavailable'))
    result = heartbeat.pulse(path, now=1000)
    assert result['status'] == 'queue_failed' and result['needs_attention']
    assert result.get('queued_sequence', 0) == 0
