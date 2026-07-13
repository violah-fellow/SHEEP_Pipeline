## Pipeline for Publication Data collection and classification
## Calls S0-S4 in sequence with checkpoint/resume support.
## To restart a completed or unwanted run, delete its status file from status_logs/.

import os
import sys
from datetime import datetime

from Helper_pipeline_functions import save_status, mark_done, find_incomplete_run
from S0_ML_training import main as train_classifiers
from S1_query_dimensions import main as query_dimensions
from S2_ML_classification import main as classify_run
from S3_LLM_scope import main as scope_llm
from S4_query_reverse import main as query_reverse

# CONFIG 
# edit parameters for this run here

# Resuming inclompete runs rom status_log
RESUME = True

# Shared paths
DB_PATH           = 'publications.db'
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'
KEY_PATH          = '../.env' # or '../../.env' if running from the pipeline folder

# Query
STRINGS_FILE = 'dimensions_search_publications.txt'
YEAR_FROM    = 2023
YEAR_TO      = 2024

# GenAI
LLM_MODEL_SCOPE = 'claude-sonnet-4-6'   # or 'claude-haiku-4-5' for cheap test runs
# path to the system prompt used for scoping
PROMPT_PATH = 'llm_prompts/scope_prompt_publications.md'

# Training
TRAINING_TABLE        = 'publications_new_training'
EMBEDDINGS_PATH_TRAIN = 'embeddings/embeddings_new_training.npy'
MAX_FN                = 0.01

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
    EMBEDDINGS_PATH_REVERSE = f'embeddings/embeddings_{REVERSE_TABLE}.npy'

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
        'THRESHOLD_PATH':         THRESHOLD_PATH,
        'LLM_MODEL_SCOPE':        LLM_MODEL_SCOPE,
        'PROMPT_PATH':            PROMPT_PATH,
        'MAX_FN':                 MAX_FN,
        'TRAINING_TABLE':         TRAINING_TABLE,
        'EMBEDDINGS_PATH_TRAIN':  EMBEDDINGS_PATH_TRAIN,
        'EMBEDDINGS_PATH_RUN':    EMBEDDINGS_PATH_RUN,
        'EMBEDDINGS_PATH_REVERSE': EMBEDDINGS_PATH_REVERSE,
    }
    status = {
        'config': cfg,
        'steps': {
            'train':            'pending',
            'query':            'pending',
            'classify':         'pending',
            'llm_scope':         'pending',
            'reverse_query':     'pending',
            'reverse_classify':  'pending',
            'reverse_llm_scope': 'pending',
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
    #     DB_PATH=cfg['DB_PATH'],
    #     TRAINING_TABLE=cfg['TRAINING_TABLE'],
    #     EMBEDDINGS_PATH=cfg['EMBEDDINGS_PATH_TRAIN'],
    #     SCOPE_MODEL_PATH=cfg['SCOPE_MODEL_PATH'],
    #     PILLAR_MODEL_PATH=cfg['PILLAR_MODEL_PATH'],
    #     THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
    #     MAX_FN=cfg['MAX_FN],
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
    )
    mark_done(status, 'classify', STATUS_DIR)
else:
    print("\nStep 2 (classify) already done, skipping.")

# Step 3: LLM scope classification
if status['steps']['llm_scope'] != 'done':
    print("\nStarting Step 3: LLM scope classification.")
    scope_llm(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
        PROMPT_PATH=cfg['PROMPT_PATH'],
    )
    mark_done(status, 'llm_scope', STATUS_DIR)
else:
    print("\nStep 3 (llm_scope) already done, skipping.")

# Step 4: Reverse search
if status['steps']['reverse_query'] != 'done':
    print("\nStarting Step 4: Reverse search for researcher IDs.")
    query_reverse(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
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
        THRESHOLD_PATH=cfg['THRESHOLD_PATH'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
    )
    mark_done(status, 'reverse_llm_scope', STATUS_DIR)
else:
    print("\nStep 6 (reverse_llm_scope) already done, skipping.")

print(f"\nPipeline complete for run '{cfg['RUN_TABLE']}'.")
