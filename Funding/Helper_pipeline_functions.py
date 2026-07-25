## Helper functions for pipeline.py checkpoint/resume system

import os, json


def status_path(status_dir, run_table):
    return os.path.join(status_dir, f'status_{run_table}.json')


def save_status(status, status_dir):
    with open(status_path(status_dir, status['config']['RUN_TABLE']), 'w') as f:
        json.dump(status, f, indent=2)


def mark_done(status, step, status_dir):
    status['steps'][step] = 'done'
    save_status(status, status_dir)


def find_incomplete_run(status_dir):
    """Return the status dict of the most recent incomplete run, or None."""
    files = sorted([
        f for f in os.listdir(status_dir)
        if f.startswith('status_') and f.endswith('.json')
    ], reverse=True)
    for fname in files:
        with open(os.path.join(status_dir, fname)) as f:
            s = json.load(f)
        if any(v != 'done' for v in s['steps'].values()):
            return s
    return None
