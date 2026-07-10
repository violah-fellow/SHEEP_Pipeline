## Pipeline for Patent Data collection and classification
## Calls S0-S3 in sequence with checkpoint/resume support.
## To restart a completed or unwanted run, delete its status file from status_logs/.

import os
import sys
from datetime import datetime

from Helper_pipeline_functions import save_status, mark_done, find_incomplete_run
from S0_ML_training import main as train_classifiers
from S1_query_dimensions import main as query_dimensions
from S2_ML_classification import main as classify_run
from S3_LLM_scope import main as llm_scope
from S4_query_reverse import main as query_reverse
from S5_Assignee_cleanup import main as assignee_cleanup

# CONFIG 
# edit parameters for this run here

# Resuming inclompete runs rom status_log
RESUME = True

# Shared paths
DB_PATH           = 'patents.db'
DB_PATH_TRAINING  = 'patents_training.db'
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'
KEY_PATH          = '../.env'

# Query
STRINGS_FILE    = 'dimensions_search_patents.txt'
CPC_SEARCH_FILE = 'CPC_for_query.txt'
CPC_FILTER_FILE = 'CPC_for_filter.txt'
YEAR            = 2025

# Training
TRAINING_TABLE        = 'patents_raw'
EMBEDDINGS_TABLE      = 'patents_embeddings'
EMBEDDINGS_PATH_TRAIN = 'embeddings/embeddings_new_training.npy'
MAX_FN                = 0.01

# Classification
CLASSIFICATION_TABLE = 'patents_classified'

# LLM scoping
LLM_MODEL_SCOPE       = 'claude-haiku-4-5'
PROMPT_PATH           = 'llm_prompts/scope_prompt_patents.md'
BATCH_DIR             = 'batch_jobs'
POLL_INTERVAL_SECONDS = 1800

# Assignee name clean-up
ASSIGNEE_COLUMN = 'assignee_names'
CLEAN_COLUMN = 'assignee_names_cleaned'
COMPANY_LIST = 'company_list.csv'
COMPANY_DATABASE = 'company_database.csv'

STATUS_DIR = 'status_logs'
LOG_DIR    = 'run_logs'
os.makedirs(STATUS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


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
incomplete = find_incomplete_run(STATUS_DIR) if RESUME else None

if incomplete:
    cfg = incomplete['config']
    status = incomplete
    _log_path = os.path.join(LOG_DIR, f"{cfg['RUN_TABLE']}.log")
    _log_file = open(_log_path, 'a', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Resuming run '{cfg['RUN_TABLE']}'"
          f"\ncompleted steps: {[k for k,v in status['steps'].items() if v == 'done']}")
else:
    date = datetime.today().strftime("%y%m%d_%H%M")
    RUN_TABLE             = f'run_{date}'
    REVERSE_TABLE         = RUN_TABLE + '_reverse'
    EMBEDDINGS_PATH_RUN     = f'embeddings/embeddings_{RUN_TABLE}.npy'

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
        'YEAR':                   YEAR,
        'SCOPE_MODEL_PATH':       SCOPE_MODEL_PATH,
        'PILLAR_MODEL_PATH':      PILLAR_MODEL_PATH,
        'THRESHOLD_PATH':         THRESHOLD_PATH,
        'MAX_FN':                 MAX_FN,
        'TRAINING_TABLE':         TRAINING_TABLE,
        'EMBEDDINGS_TABLE':       EMBEDDINGS_TABLE,
        'EMBEDDINGS_PATH_TRAIN':  EMBEDDINGS_PATH_TRAIN,
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
    status = {
        'config': cfg,
        'steps': {
            'train':            'pending',
            'query':            'pending',
            'classify':         'pending',
            'llm_scope':        'pending',
            'reverse_query':    'pending',
            'assignee_cleanup': 'pending',
        }
    }
    save_status(status, STATUS_DIR)
    _log_path = os.path.join(LOG_DIR, f"{RUN_TABLE}.log")
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Starting new run '{cfg['RUN_TABLE']}'")


# Step 0: Train classifiers
if status['steps']['train'] != 'done':
    print("\nStarting Step 0: Train classifiers.")
    # train_classifiers(
    #     DB_PATH=cfg['DB_PATH_TRAINING'],
    #     TRAINING_TABLE=cfg['TRAINING_TABLE'],
    #     EMBEDDINGS_TABLE=cfg['EMBEDDINGS_TABLE'],
    #     EMBEDDINGS_PATH=cfg['EMBEDDINGS_PATH_TRAIN'],
    #     SCOPE_MODEL_PATH=cfg['SCOPE_MODEL_PATH'],
    #     PILLAR_MODEL_PATH=cfg['PILLAR_MODEL_PATH'],
    #     THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
    #     MAX_FN=cfg['MAX_FN'],
    # )
    mark_done(status, 'train', STATUS_DIR)
else:
    print("\nStep 0 (train) already done, skipping.")

# Step 1: Query Dimensions
if status['steps']['query'] != 'done':
    print("\nStarting Step 1: Query Dimensions for publications.")
    query_dimensions(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        STRINGS_FILE=cfg['STRINGS_FILE'],
        CPC_SEARCH_FILE=cfg['CPC_SEARCH_FILE'],
        CPC_FILTER_FILE=cfg['CPC_FILTER_FILE'],
        YEAR=cfg['YEAR'],
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
if status['steps']['llm_scope'] != 'done':
    print("\nStarting Step 3: LLM scoping.")
    llm_scope(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
        PROMPT_PATH=cfg['PROMPT_PATH'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
        BATCH_DIR=cfg['BATCH_DIR'],
        POLL_INTERVAL_SECONDS=cfg['POLL_INTERVAL_SECONDS'],
    )
    mark_done(status, 'llm_scope', STATUS_DIR)
else:
    print("\nStep 3 (llm_scope) already done, skipping.")

# Step 4: Reverse search
if status['steps']['reverse_query'] != 'done':
    print("\nStarting Step 4: Reverse search for family members.")
    query_reverse(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        REVERSE_TABLE=cfg['REVERSE_TABLE'],
        YEAR=cfg['YEAR'],
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

