## Pipeline for Publication Data collection and classification
## Calls S0-S4 in sequence with checkpoint/resume support.
## (more precisely S1-S4: S0_ML_training.py is not part of a run - the models are trained separately)
##
## THIS IS THE SCRIPT TO START for a normal data collection run:
##     python pipeline_publications.py
##
## What it does: sets the parameters for one run, then executes the step scripts in order and records
## after each step that it is finished. It runs six steps - the same three steps twice, once for the
## keyword search and once for the reverse search:
##     Step 1  S1_query_dimensions.py   query Dimensions with the search strings   -> RUN_TABLE
##     Step 2  S2_ML_classification.py  ML pre-filter (embeddings + classifiers)
##     Step 3  S3_LLM_scope.py          LLM scope/pillar check (can take hours)
##     Step 4  S4_query_reverse.py      reverse search via researchers             -> REVERSE_TABLE
##     Step 5  S2_ML_classification.py  ML pre-filter for the reverse table
##     Step 6  S3_LLM_scope.py          LLM scope/pillar check for the reverse table
##
## What happens AFTER this script (both by hand, not part of the pipeline):
##     S5_Review_classifications.ipynb  manual review of the borderline cases
##     S6_LLM_labelling.py              research-category labelling of the reviewed records
##     S7_Retrieve_dataset.ipynb        export the final dataset as CSV
##
## Resume: every run writes status_logs/status_<RUN_TABLE>.json. If the script is interrupted (or a
## step fails), simply start it again - completed steps are skipped and it continues where it stopped.
## Since step 3 and step 6 wait for a batch that can take hours, this is the normal way to work: let
## it run, interrupt it, start it again later.
## Beware: a resumed run uses the config stored in its status file, not the CONFIG below. See
## Helper_pipeline_functions.py for details.
## To restart a completed or unwanted run, delete its status file from status_logs/.
##
## Console output is written both to the terminal and to run_logs/<RUN_TABLE>.log (see _Tee below).

import os
import sys
from datetime import datetime

# the step scripts are imported and their main() called - so they run in the same process and their
# printed output ends up in this run's log file
from Helper_pipeline_functions import save_status, mark_done, find_incomplete_run
from S1_query_dimensions import main as query_dimensions
from S2_ML_classification import main as classify_run
from S3_LLM_scope import main as scope_llm
from S4_query_reverse import main as query_reverse

# CONFIG 
# edit parameters for this run here

# !! These values only take effect for a NEW run. They are copied into the run's status file when it
# !! starts, and a resumed run reads them from there - editing them here does nothing for a run that
# !! is already in progress (edit status_logs/status_<RUN_TABLE>.json, or delete it to start over).

# Resuming inclompete runs rom status_log   [resuming incomplete runs from status_logs]
# True  = pick up the most recent unfinished run (the normal case)
# False = always start a new run, even if an unfinished one exists (its tables stay in the database)
RESUME = True

# Shared paths
DB_PATH           = 'publications.db'
# the model files trained by S0_ML_training.py; all three must come from the same training run
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'
# the .env file with DIMENSIONS_API_KEY and CLAUDE_API_KEY; paths are relative to the folder you
# start the script from, which is why this needs adjusting when running from elsewhere
KEY_PATH          = '../.env' # or '../../.env' if running from the pipeline folder

# Query
STRINGS_FILE = 'dimensions_search_publications.txt'
# publication years for BOTH the keyword search and the reverse search (inclusive on both ends).
# This is the biggest cost lever of a run: a wide range means many records, and each of them is
# embedded (step 2/5) and sent to the LLM (step 3/6). For a first test, use a single year.
YEAR_FROM    = 2010
YEAR_TO      = 2022

# Classification
# permanent embedding cache (nothing is embedded twice) ...
EMBEDDINGS_TABLE     = 'publications_embeddings'
# ... and the main table all runs accumulate into. Both live in DB_PATH.
CLASSIFICATION_TABLE = 'publications_classified'

# GenAI
LLM_MODEL_SCOPE = 'claude-sonnet-4-6'   # or 'claude-haiku-4-5' for cheap test runs
# path to the system prompt used for scoping
# contains the scope definition - the file to edit when what counts as "in scope" changes
PROMPT_PATH = 'llm_prompts/scope_prompt_publications.md'
# directory for batch submission metadata, keyed by RUN_TABLE (allows resuming without resubmitting)
BATCH_DIR = 'batch_jobs'
# how often to check whether a batch has finished
# 1800 s = 30 min; steps 3 and 6 spend most of their time in this waiting loop
POLL_INTERVAL_SECONDS = 1800

# status_logs = which steps of which run are finished (resume information)
# run_logs    = full console output per run
STATUS_DIR = 'status_logs'
LOG_DIR    = 'run_logs'
os.makedirs(STATUS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# small helper that writes everything printed to several targets at once (terminal + log file).
# Used below via sys.stdout = _Tee(...), so no print() call in this script or the step scripts needs
# to change. flush() after every write means the log file is up to date even if the run is killed.
class _Tee:
    def __init__(self, *streams):
        self._streams = streams
    def write(self, data):
        for s in self._streams:
            s.write(data)
            s.flush()
    def flush(self):
        for s in self._streams:
            s.flush()


# START SCRIPT

# Resolve config: resume incomplete run or start fresh
# everything below happens at import time, i.e. simply by starting the script - there is no main()
# function here. The two branches decide whether an existing run is continued or a new one begins.
incomplete = find_incomplete_run(STATUS_DIR) if RESUME else None

if incomplete:
    # RESUME CASE: config and step states come from the status file, nothing is reset
    cfg = incomplete['config']
    status = incomplete
    # 'a' = append, so the log of the existing run is continued rather than overwritten
    _log_path = os.path.join(LOG_DIR, f"{cfg['RUN_TABLE']}.log")
    _log_file = open(_log_path, 'a', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Resuming run '{cfg['RUN_TABLE']}'"
          f"\ncompleted steps: {[k for k,v in status['steps'].items() if v == 'done']}")
else:
    # NEW RUN: the run name is the current date and time, which makes every table, log and embedding
    # file of this run recognisable and keeps runs from overwriting each other
    date = datetime.today().strftime("%y%m%d_%H%M")
    RUN_TABLE             = f'run_{date}'
    REVERSE_TABLE         = RUN_TABLE + '_reverse'
    # separate checkpoint file per table - important, because the embedding checkpoint counts rows and
    # must not be shared between two different sets of records (see get_embeddings())
    EMBEDDINGS_PATH_RUN     = f'embeddings/embeddings_{RUN_TABLE}.npy'
    EMBEDDINGS_PATH_REVERSE = f'embeddings/embeddings_{REVERSE_TABLE}.npy'

    # this dict is the run's complete configuration; it is saved to the status file and is what a
    # resumed run reads back. Anything not in here cannot be reconstructed later.
    cfg = {
        'DB_PATH':                DB_PATH,
        'KEY_PATH':               KEY_PATH,
        'RUN_TABLE':              RUN_TABLE,
        'REVERSE_TABLE':          REVERSE_TABLE,
        'STRINGS_FILE':           STRINGS_FILE,
        'YEAR_FROM':              YEAR_FROM,
        'YEAR_TO':                YEAR_TO,
        'SCOPE_MODEL_PATH':       SCOPE_MODEL_PATH,
        'PILLAR_MODEL_PATH':      PILLAR_MODEL_PATH,
        'EMBEDDINGS_TABLE':       EMBEDDINGS_TABLE,
        'CLASSIFICATION_TABLE':   CLASSIFICATION_TABLE,
        'THRESHOLD_PATH':         THRESHOLD_PATH,
        'LLM_MODEL_SCOPE':        LLM_MODEL_SCOPE,
        'PROMPT_PATH':            PROMPT_PATH,
        'BATCH_DIR':              BATCH_DIR,
        'POLL_INTERVAL_SECONDS':  POLL_INTERVAL_SECONDS,
        'EMBEDDINGS_PATH_RUN':    EMBEDDINGS_PATH_RUN,
        'EMBEDDINGS_PATH_REVERSE': EMBEDDINGS_PATH_REVERSE,
    }
    # all steps start as 'pending' and are set to 'done' by mark_done() after they finish. The keys
    # are the step names used in the queries below - they must match.
    status = {
        'config': cfg,
        'steps': {
            'query':            'pending',
            'classify':         'pending',
            'llm_scope':         'pending',
            'reverse_query':     'pending',
            'reverse_classify':  'pending',
            'reverse_llm_scope': 'pending',
        }
    }
    # written immediately, so even a run that crashes in step 1 leaves a resumable status file
    save_status(status, STATUS_DIR)
    _log_path = os.path.join(LOG_DIR, f"{RUN_TABLE}.log")
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Starting new run '{cfg['RUN_TABLE']}'")

# every step follows the same pattern: skip if already 'done', otherwise call the step script's main()
# and mark it done afterwards. A step that raises stops the script - nothing is marked, so the next
# start repeats exactly that step.

# Step 1: Query Dimensions
if status['steps']['query'] != 'done':
    print("\nStarting Step 1: Query Dimensions for publications.")
    query_dimensions(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        STRINGS_FILE=cfg['STRINGS_FILE'],
        YEAR_FROM=cfg['YEAR_FROM'],
        YEAR_TO=cfg['YEAR_TO'],
    )
    mark_done(status, 'query', STATUS_DIR)
else:
    print("\nStep 1 (query) already done, skipping.")

# Step 2: Classify queried publications with ML
if status['steps']['classify'] != 'done':
    print("\nStarting Step 2: Classify queried publications.")
    classify_run(
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        EMBEDDINGS_PATH=cfg['EMBEDDINGS_PATH_RUN'],
        SCOPE_MODEL_PATH=cfg['SCOPE_MODEL_PATH'],
        PILLAR_MODEL_PATH=cfg['PILLAR_MODEL_PATH'],
        THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
        EMBEDDINGS_TABLE=cfg['EMBEDDINGS_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
    )
    mark_done(status, 'classify', STATUS_DIR)
else:
    print("\nStep 2 (classify) already done, skipping.")

# Step 3: LLM scope classification
# this is the long step: it submits a batch to Claude and then waits for the result (possibly hours).
# Interrupting is harmless - the batch keeps running server-side and is collected on the next start.
if status['steps']['llm_scope'] != 'done':
    print("\nStarting Step 3: LLM scope classification.")
    scope_llm(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
        PROMPT_PATH=cfg['PROMPT_PATH'],
        BATCH_DIR=cfg['BATCH_DIR'],
        POLL_INTERVAL_SECONDS=cfg['POLL_INTERVAL_SECONDS'],
    )
    mark_done(status, 'llm_scope', STATUS_DIR)
else:
    print("\nStep 3 (llm_scope) already done, skipping.")

# Step 4: Reverse search
# steps 4-6 repeat the same logic for the reverse search: query researchers -> ML -> LLM. Steps 5 and
# 6 call the same functions as steps 2 and 3, only with REVERSE_TABLE instead of RUN_TABLE.
if status['steps']['reverse_query'] != 'done':
    print("\nStarting Step 4: Reverse search for researcher IDs.")
    query_reverse(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        REVERSE_TABLE=cfg['REVERSE_TABLE'],
        YEAR_FROM=cfg['YEAR_FROM'],
        YEAR_TO=cfg['YEAR_TO'],
    )
    mark_done(status, 'reverse_query', STATUS_DIR)
else:
    print("\nStep 4 (reverse_query) already done, skipping.")

# Step 5: Classify reverse search publications with ML
if status['steps']['reverse_classify'] != 'done':
    print("\nStarting Step 5: Classify reverse search publications.")
    classify_run(
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['REVERSE_TABLE'],
        EMBEDDINGS_PATH=cfg['EMBEDDINGS_PATH_REVERSE'],
        SCOPE_MODEL_PATH=cfg['SCOPE_MODEL_PATH'],
        PILLAR_MODEL_PATH=cfg['PILLAR_MODEL_PATH'],
        EMBEDDINGS_TABLE=cfg['EMBEDDINGS_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
    )
    mark_done(status, 'reverse_classify', STATUS_DIR)
else:
    print("\nStep 5 (reverse_classify) already done, skipping.")

# Step 6: LLM scope classification for reverse search publications
if status['steps']['reverse_llm_scope'] != 'done':
    print("\nStarting Step 6: LLM scope classification for reverse search publications.")
    scope_llm(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['REVERSE_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
        PROMPT_PATH=cfg['PROMPT_PATH'],
        BATCH_DIR=cfg['BATCH_DIR'],
        POLL_INTERVAL_SECONDS=cfg['POLL_INTERVAL_SECONDS'],
    )
    mark_done(status, 'reverse_llm_scope', STATUS_DIR)
else:
    print("\nStep 6 (reverse_llm_scope) already done, skipping.")

print(f"\nPipeline complete for run '{cfg['RUN_TABLE']}'.")
# What comes next (by hand): S5_Review_classifications.ipynb (manual review) ->
# S6_LLM_labelling.py (research categories) -> S7_Retrieve_dataset.ipynb (CSV export)
