"""Queue same-thread progress checks from an OS scheduler, outside agent tool cells.

Uses the installed Codex CLI's public queue command. One unacknowledged message
at most; a successful enqueue is not evidence that an agent inspected training.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
import subprocess
import time


def save(path, state):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(state, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def pulse(path, *, ack=False, disable=False, note=None, now=None):
    now = time.time() if now is None else now
    stamp = datetime.fromtimestamp(now, timezone.utc).isoformat()
    with path.with_suffix('.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        state = json.loads(path.read_text())
        if disable:
            state.update(enabled=False, status='disabled', disabled_at=stamp)
        elif ack:
            state.update(acknowledged_sequence=state.get('queued_sequence', 0),
                         last_ack_at=stamp, status='acknowledged', needs_attention=False)
            if note is not None:
                state.update(last_inspection_at=stamp, last_inspection_note=note)
        elif not state.get('enabled', False):
            return state
        elif state.get('queued_sequence', 0) > state.get('acknowledged_sequence', 0):
            state.update(status='awaiting_agent_ack', last_poll_at=stamp,
                         needs_attention=now-state['last_queued_unix'] > 900)
        else:
            sequence = state.get('queued_sequence', 0) + 1
            prompt = f"[ORCA_RL_HEARTBEAT sequence={sequence}] " + state['prompt']
            try:
                process = subprocess.run([state['codex'], 'queue', '--thread', state['threadId'],
                                          '--message', prompt], cwd=state['cwd'],
                                         capture_output=True, text=True, timeout=30)
                if process.returncode:
                    raise RuntimeError(f'Codex queue exited {process.returncode}: {process.stderr[-1000:]}')
                state.update(queued_sequence=sequence, last_queued_at=stamp, last_queued_unix=now,
                             status='queued', needs_attention=False, queue_receipt=process.stdout.strip())
            except (OSError, RuntimeError, subprocess.TimeoutExpired) as error:
                state.update(status='queue_failed', needs_attention=True, last_error=str(error))
        save(path, state)
        event = {k: state.get(k) for k in ('status', 'queued_sequence', 'acknowledged_sequence',
                                          'needs_attention', 'last_inspection_at')}
        event['timestamp'] = stamp
        with path.with_suffix('.jsonl').open('a') as log:
            log.write(json.dumps(event) + '\n')
        print(json.dumps(event), flush=True)
        return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--ack', action='store_true')
    mode.add_argument('--disable', action='store_true')
    parser.add_argument('--note', help='Inspection evidence recorded only with --ack')
    args = parser.parse_args()
    if args.note is not None and not args.ack:
        parser.error('--note requires --ack')
    pulse(args.state.resolve(), ack=args.ack, disable=args.disable, note=args.note)


if __name__ == '__main__':
    main()
