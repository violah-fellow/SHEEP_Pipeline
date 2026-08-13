## Pipeline for Patent Data collection and classification
## Calls S0-S3 in sequence with checkpoint/resume support.
## (more precisely S1-S5: S0_ML_training.py is not part of a run - the models are trained separately)
##
## THIS IS THE SCRIPT TO START for a normal data collection run:
##     python pipeline_patents.py
##
## What it does: sets the parameters for one run, then executes the step scripts in order and records
## after each step that it is finished:
##     Step 1  S1_query_dimensions.py   query Dimensions (keywords + CPC codes)  -> RUN_TABLE
##     Step 2  S2_ML_classification.py  ML pre-filter, one representative per family
##     Step 3  S3_LLM_scope.py          LLM scope/pillar check (can take hours)
##     Step 4  S4_query_reverse.py      fetch remaining family members, inheriting the predictions
##     Step 5  S5_Assignee_cleanup.py   normalise company names
##
## What happens AFTER this script (both by hand, not part of the pipeline):
##     S6_Review_classifications.ipynb  manual review of the borderline cases
##     S7_LLM_labelling.py              the four detailed label types
##     S8_Retrieve_dataset.ipynb        export the final dataset as CSV
##
## Unlike the publications pipeline the reverse search runs only ONCE and needs no ML/LLM steps of its
## own - family members inherit the classifications of their already scored siblings.
##
## Resume: every run writes status_logs/status_<RUN_TABLE>.json. If the script is interrupted (or a
## step fails), simply start it again - completed steps are skipped. Since step 3 waits for a batch
## that can take hours, this is the normal way to work.
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
from S3_LLM_scope import main as llm_scope
from S4_query_reverse import main as query_reverse
from S5_Assignee_cleanup import main as assignee_cleanup

# CONFIG 
# edit parameters for this run here

# !! These values only take effect for a NEW run. They are copied into the run's status file when it
# !! starts, and a resumed run reads them from there - editing them here does nothing for a run that
# !! is already in progress (edit status_logs/status_<RUN_TABLE>.json, or delete it to start over).

# Resuming inclompete runs rom status_log   [resuming incomplete runs from status_logs]
# True  = pick up the most recent unfinished run (the normal case)
# False = always start a new run, even if an unfinished one exists
RESUME = True

# Shared paths
DB_PATH           = 'patents.db'
# only used to build the config; the training database is not touched during a run
DB_PATH_TRAINING  = 'patents_training.db'
# the model files trained by S0_ML_training.py; all three must come from the same training run
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'
# the .env file with DIMENSIONS_API_KEY and CLAUDE_API_KEY, relative to the folder you start from
KEY_PATH          = '../.env'

# Query
# search strings, CPC codes to search, and the narrower CPC list used to filter afterwards
STRINGS_FILE    = 'dimensions_search_patents.txt'
CPC_SEARCH_FILE = 'CPC_for_query.txt'
CPC_FILTER_FILE = 'CPC_for_filter.txt'
# publication years, inclusive on both ends. The biggest cost lever of a run - and it also limits
# which family members step 4 can find at all.
YEAR_FROM       = 2010
YEAR_TO         = 2022

# Classification
# embedding cache (representatives only) and the main table all runs accumulate into
EMBEDDINGS_TABLE     = 'patents_embeddings'
CLASSIFICATION_TABLE = 'patents_classified'

# LLM scoping
# sonnet for real runs; 'claude-haiku-4-5' is markedly cheaper for testing
LLM_MODEL_SCOPE       = 'claude-sonnet-4-6'
# contains the scope definition - the file to edit when what counts as "in scope" changes
PROMPT_PATH           = 'llm_prompts/scope_prompt_patents.md'
# batch metadata per run; needed to resume without resubmitting
BATCH_DIR             = 'batch_jobs'
# 1800 s = 30 min between checks; step 3 spends most of its time in this waiting loop
POLL_INTERVAL_SECONDS = 1800

# Assignee name clean-up
# source column (as delivered by Dimensions) and the new normalised column
ASSIGNEE_COLUMN = 'assignee_names'
CLEAN_COLUMN = 'assignee_names_cleaned'
# reference data for the clean-up - see the note on hardcoded column names in S5_Assignee_cleanup.py
COMPANY_LIST = 'company_list.csv'
COMPANY_DATABASE = 'company_database.csv'

# status_logs = which steps of which run are finished (resume information)
# run_logs    = full console output per run
STATUS_DIR = 'status_logs'
LOG_DIR    = 'run_logs'
os.makedirs(STATUS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# small helper that writes everything printed to several targets at once (terminal + log file).
# Used below via sys.stdout = _Tee(...), so no print() call has to change. flush() after every write
# means the log file is up to date even if the run is killed.
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
# everything below runs simply by starting the script - there is no main() function here. The two
# branches decide whether an existing run is continued or a new one begins.
incomplete = find_incomplete_run(STATUS_DIR) if RESUME else None

if incomplete:
    # RESUME CASE: config and step states come from the status file, nothing is reset
    cfg = incomplete['config']
    status = incomplete
    # 'a' = append, so the existing run's log is continued rather than overwritten
    _log_path = os.path.join(LOG_DIR, f"{cfg['RUN_TABLE']}.log")
    _log_file = open(_log_path, 'a', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Resuming run '{cfg['RUN_TABLE']}'"
          f"\ncompleted steps: {[k for k,v in status['steps'].items() if v == 'done']}")
else:
    # NEW RUN: the run name is the current date and time, so every table, log and embedding file of
    # this run is recognisable and runs never overwrite each other
    date = datetime.today().strftime("%y%m%d_%H%M")
    RUN_TABLE             = f'run_{date}'
    REVERSE_TABLE         = RUN_TABLE + '_reverse'
    # own checkpoint file per run - the embedding checkpoint counts rows and must not be shared
    # between different sets of records (see get_embeddings()). Only one is needed here, because the
    # reverse table is not embedded.
    EMBEDDINGS_PATH_RUN     = f'embeddings/embeddings_{RUN_TABLE}.npy'

    # this dict is the run's complete configuration; it is saved to the status file and is what a
    # resumed run reads back. Anything not in here cannot be reconstructed later.
    cfg = {
        'DB_PATH':                DB_PATH,
        'DB_PATH_TRAINING':       DB_PATH_TRAINING,
        'KEY_PATH':               KEY_PATH,
        'RUN_TABLE':              RUN_TABLE,
        'REVERSE_TABLE':          REVERSE_TABLE,
        'CLASSIFICATION_TABLE':   CLASSIFICATION_TABLE,
        'STRINGS_FILE':           STRINGS_FILE,
        'CPC_SEARCH_FILE':        CPC_SEARCH_FILE,
        'CPC_FILTER_FILE':        CPC_FILTER_FILE,
        'YEAR_FROM':              YEAR_FROM,
        'YEAR_TO':                YEAR_TO,
        'SCOPE_MODEL_PATH':       SCOPE_MODEL_PATH,
        'PILLAR_MODEL_PATH':      PILLAR_MODEL_PATH,
        'THRESHOLD_PATH':         THRESHOLD_PATH,
        'EMBEDDINGS_TABLE':       EMBEDDINGS_TABLE,
        'EMBEDDINGS_PATH_RUN':    EMBEDDINGS_PATH_RUN,
        'LLM_MODEL_SCOPE':        LLM_MODEL_SCOPE,
        'PROMPT_PATH':            PROMPT_PATH,
        'BATCH_DIR':              BATCH_DIR,
        'POLL_INTERVAL_SECONDS':  POLL_INTERVAL_SECONDS,
        'ASSIGNEE_COLUMN':        ASSIGNEE_COLUMN,
        'CLEAN_COLUMN':           CLEAN_COLUMN,
        'COMPANY_LIST':           COMPANY_LIST,
        'COMPANY_DATABASE':       COMPANY_DATABASE,
    }
    # all steps start as 'pending' and are set to 'done' by mark_done() once finished. The keys must
    # match the ones queried further down.
    status = {
        'config': cfg,
        'steps': {
            'query':            'pending',
            'classify':         'pending',
            'llm_scope':        'pending',
            'reverse_query':    'pending',
            'assignee_cleanup': 'pending',
        }
    }
    # written immediately, so even a run that crashes in step 1 leaves a resumable status file
    save_status(status, STATUS_DIR)
    _log_path = os.path.join(LOG_DIR, f"{RUN_TABLE}.log")
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Starting new run '{cfg['RUN_TABLE']}'")


# every step follows the same pattern: skip if already 'done', otherwise call the step script's
# main() and mark it done afterwards. A step that raises stops the script - nothing is marked, so the
# next start repeats exactly that step.

# Step 1: Query Dimensions
if status['steps']['query'] != 'done':
    # (the messages in this script say "publications" in a few places - it means patents)
    print("\nStarting Step 1: Query Dimensions for publications.")
    query_dimensions(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        STRINGS_FILE=cfg['STRINGS_FILE'],
        CPC_SEARCH_FILE=cfg['CPC_SEARCH_FILE'],
        CPC_FILTER_FILE=cfg['CPC_FILTER_FILE'],
        YEAR_FROM=cfg['YEAR_FROM'],
        YEAR_TO=cfg['YEAR_TO'],
    )
    mark_done(status, 'query', STATUS_DIR)
else:
    print("\nStep 1 (query) already done, skipping.")

# Step 2: Classify queried publications
if status['steps']['classify'] != 'done':
    print("\nStarting Step 2: Classify queried publications.")
    classify_run(
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        EMBEDDINGS_TABLE=cfg['EMBEDDINGS_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        EMBEDDINGS_PATH=cfg['EMBEDDINGS_PATH_RUN'],
        SCOPE_MODEL_PATH=cfg['SCOPE_MODEL_PATH'],
        PILLAR_MODEL_PATH=cfg['PILLAR_MODEL_PATH'],
        THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
    )
    mark_done(status, 'classify', STATUS_DIR)
else:
    print("\nStep 2 (classify) already done, skipping.")

# Step 3: LLM scoping
# the long step: submits a batch to Claude and then waits for the result (possibly hours).
# Interrupting is harmless - the batch keeps running server-side and is collected on the next start.
if status['steps']['llm_scope'] != 'done':
    print("\nStarting Step 3: LLM scoping.")
    llm_scope(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        PROMPT_PATH=cfg['PROMPT_PATH'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
        BATCH_DIR=cfg['BATCH_DIR'],
        POLL_INTERVAL_SECONDS=cfg['POLL_INTERVAL_SECONDS'],
    )
    mark_done(status, 'llm_scope', STATUS_DIR)
else:
    print("\nStep 3 (llm_scope) already done, skipping.")

# Step 4: Reverse search
# fetches the remaining members of the in-scope families. They inherit the ML and LLM results of
# their family, so no further classification steps are needed here.
if status['steps']['reverse_query'] != 'done':
    print("\nStarting Step 4: Reverse search for family members.")
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

# Step 5: Assignee clean-up
if status['steps']['assignee_cleanup'] != 'done':
    print("\nStarting Step 5: Clean-up of assignee names.")
    assignee_cleanup(
        DB_PATH=cfg['DB_PATH'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        ASSIGNEE_COLUMN=cfg['ASSIGNEE_COLUMN'],
        CLEAN_COLUMN=cfg['CLEAN_COLUMN'],
        COMPANY_LIST=cfg['COMPANY_LIST'],
        COMPANY_DATABASE=cfg['COMPANY_DATABASE'],
    )
    mark_done(status, 'assignee_cleanup', STATUS_DIR)
else:
    print("\nStep 5 (assignee_cleanup) already done, skipping.")

# Note: unlike pipeline_publications.py this script prints no closing message - once step 5 is done
# the run is complete.
# What comes next (by hand): S6_Review_classifications.ipynb (manual review) ->
# S7_LLM_labelling.py (the four label types) -> S8_Retrieve_dataset.ipynb (CSV export)

