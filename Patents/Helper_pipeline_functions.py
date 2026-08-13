## Helper functions for the checkpoint/resume system used by the pipeline scripts
## (pipeline_publications.py / pipeline_patents.py / pipeline_funding.py).
## This file is identical in all three pipeline folders - keep it that way if you change it.
##
## How the checkpointing works: every pipeline run writes one small JSON file into status_logs/,
## named status_<RUN_TABLE>.json, holding
##   'config' -> the complete set of parameters that run was started with
##   'steps'  -> one entry per pipeline step, either 'pending' or 'done'
## The pipeline script reads that file at start-up, so a run that stopped halfway (a crash, a
## closed terminal, or one of the manual notebook steps in between) picks up where it left off
## instead of re-querying and re-classifying everything.
##
## Two consequences worth knowing before you rely on this:
## - A resumed run uses the config stored in ITS OWN status file, not the CONFIG block at the top
##   of the pipeline script. Editing the script has no effect on a run that already started - to
##   change a setting mid-run, edit the JSON in status_logs/ directly.
## - To make a run start over from step 1, delete its status_<RUN_TABLE>.json. Note that this only
##   resets the bookkeeping; tables the run already wrote to the database stay as they are.

import os, json


## status_path
## builds the path of one run's status file; RUN_TABLE is the unique name of the run
## (e.g. 'run_260710_1615'), so each run gets its own file and runs never overwrite each other
def status_path(status_dir, run_table):
    return os.path.join(status_dir, f'status_{run_table}.json')


## save_status
## writes the whole status dict (config + step states) back to disk
## called after every completed step, so an interrupted run never loses more than the step it was in
def save_status(status, status_dir):
    with open(status_path(status_dir, status['config']['RUN_TABLE']), 'w') as f:
        json.dump(status, f, indent=2)


## mark_done
## flags one step as finished and immediately persists it
## step = the key used in the pipeline script's 'steps' dict, e.g. 'query', 'classify', 'llm_scope'
def mark_done(status, step, status_dir):
    status['steps'][step] = 'done'
    save_status(status, status_dir)


## find_incomplete_run
## looks for a run that was started but not finished, so the pipeline script can resume it
## returns the full status dict (config + steps) of that run, or None if every run is complete
def find_incomplete_run(status_dir):
    """Return the status dict of the most recent incomplete run, or None."""
    # sorted in reverse so the newest run is checked first: status file names contain the run
    # timestamp (status_run_260710_1615.json), which sorts chronologically as plain text
    # note: only the FIRST incomplete run found is returned. If several runs were left unfinished,
    # the newest one is resumed and the older ones stay untouched until it is finished or deleted.
    files = sorted([
        f for f in os.listdir(status_dir)
        if f.startswith('status_') and f.endswith('.json')
    ], reverse=True)
    for fname in files:
        with open(os.path.join(status_dir, fname)) as f:
            s = json.load(f)
        # "incomplete" simply means at least one step is not 'done' yet
        if any(v != 'done' for v in s['steps'].values()):
            return s
    return None
