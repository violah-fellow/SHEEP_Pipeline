## Pipeline for Funding/Grants data collection, deduplication, scoping, and labelling.
## Calls S1, S3, and S6 in sequence with checkpoint/resume support; S2 (deduplication) and S5
## (manual scope/pillar review) are manual notebook steps run by hand between them - see the
## printed instructions when the pipeline reaches those steps.
## To restart a completed or unwanted run, delete its status file from status_logs/.

import os
import sys
import duckdb
from datetime import datetime

from Helper_pipeline_functions import save_status, mark_done, find_incomplete_run
from S1_query_dimensions import main as query_dimensions
from S3_LLM_scope import main as scope_llm
#from S6_LLM_labelling import main as label_research_category

# CONFIG
# edit parameters for this run here

# Resuming incomplete runs from status_log
RESUME = True

# Shared paths
DB_PATH  = 'funding.db'  # self-contained in Pipeline/Funding, mirrors Publications
KEY_PATH = '../.env'

# Query
STRINGS_FILE = 'dimensions_search_funding.txt'
# range of start years to search across (inclusive on both ends) - set equal for a single year
START_YEAR   = 2020
END_YEAR     = 2025

# GenAI
LLM_MODEL_SCOPE = 'claude-sonnet-4-6'   # or 'claude-haiku-4-5' for cheap test runs
# path to the system prompt used for scoping
PROMPT_PATH = 'llm_prompts/scope_prompt_funding.md'
# model used for research-category labelling (S6) - PROMPT_PATHS/PILLAR_CATS/etc. are static,
# not per-run, so they're left to S6's own defaults rather than parameterized here
LLM_MODEL_LABEL = 'claude-sonnet-4-6'   # or 'claude-haiku-4-5' for cheap test runs

# Classification table (pipeline-owned, accumulates across runs)
CLASSIFICATION_TABLE = 'funding_classified'

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
    RUN_TABLE   = f'run_{date}'
    DEDUP_TABLE = RUN_TABLE + '_dedup'

    cfg = {
        'DB_PATH':               DB_PATH,
        'KEY_PATH':              KEY_PATH,
        'RUN_TABLE':             RUN_TABLE,
        'DEDUP_TABLE':           DEDUP_TABLE,
        'STRINGS_FILE':          STRINGS_FILE,
        'START_YEAR':            START_YEAR,
        'END_YEAR':              END_YEAR,
        'LLM_MODEL_SCOPE':       LLM_MODEL_SCOPE,
        'PROMPT_PATH':           PROMPT_PATH,
        'LLM_MODEL_LABEL':       LLM_MODEL_LABEL,
        'CLASSIFICATION_TABLE':  CLASSIFICATION_TABLE,
    }
    status = {
        'config': cfg,
        'steps': {
            'query':         'pending',
            'dedup':         'pending',
            'llm_scope':     'pending',
            'review':        'pending',
            'llm_labelling': 'pending',
        }
    }
    save_status(status, STATUS_DIR)
    _log_path = os.path.join(LOG_DIR, f"{RUN_TABLE}.log")
    _log_file = open(_log_path, 'w', encoding='utf-8')
    sys.stdout = sys.stderr = _Tee(sys.__stdout__, _log_file)
    print(f"Starting new run '{cfg['RUN_TABLE']}'")


# Step 1: Query Dimensions for grants
if status['steps']['query'] != 'done':
    print("\nStarting Step 1: Query Dimensions for grants.")
    query_dimensions(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        STRINGS_FILE=cfg['STRINGS_FILE'],
        START_YEAR=cfg['START_YEAR'],
        END_YEAR=cfg['END_YEAR'],
    )
    mark_done(status, 'query', STATUS_DIR)
else:
    print("\nStep 1 (query) already done, skipping.")

# Step 2: Deduplicate against tracked/historical grants (manual notebook step)
if status['steps']['dedup'] != 'done':
    db = duckdb.connect(database=cfg['DB_PATH'])
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    db.close()
    if cfg['DEDUP_TABLE'] in existing_tables:
        print(f"\nFound '{cfg['DEDUP_TABLE']}' - dedup step complete, continuing.")
        mark_done(status, 'dedup', STATUS_DIR)
    else:
        print(f"\nStep 2 (dedup) requires a manual step:"
              f"\n  1. Run S2_grant_deduplication.ipynb with RUN_TABLE='{cfg['RUN_TABLE']}' as input."
              f"\n  2. Confirm its output is saved as table '{cfg['DEDUP_TABLE']}' in {cfg['DB_PATH']}."
              f"\n  3. Re-run this script to continue (it will pick up from here).")
        sys.exit(0)
else:
    print("\nStep 2 (dedup) already done, skipping.")

# Step 3: LLM scope classification
if status['steps']['llm_scope'] != 'done':
    print("\nStarting Step 3: LLM scope classification.")
    scope_llm(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        RUN_TABLE=cfg['RUN_TABLE'],
        DEDUP_TABLE=cfg['DEDUP_TABLE'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        PROMPT_PATH=cfg['PROMPT_PATH'],
        LLM_MODEL_SCOPE=cfg['LLM_MODEL_SCOPE'],
    )
    mark_done(status, 'llm_scope', STATUS_DIR)
else:
    print("\nStep 3 (llm_scope) already done, skipping.")

# Step 4: Manual scope/pillar review (S5, manual notebook)
if status['steps']['review'] != 'done':
    db = duckdb.connect(database=cfg['DB_PATH'])
    existing_cols = db.sql(f"SELECT * FROM {cfg['CLASSIFICATION_TABLE']} LIMIT 0").df().columns.tolist()
    if 'scope_curated' not in existing_cols:
        remaining = None  # S5 has never been run at all yet
    else:
        remaining = db.sql(f"""
            SELECT COUNT(*) AS n FROM {cfg['CLASSIFICATION_TABLE']}
            WHERE "Grant ID" IN (SELECT "Grant ID" FROM {cfg['DEDUP_TABLE']})
            AND scope_curated IS NULL
        """).df()['n'][0]
    db.close()
    if remaining == 0:
        print(f"\nAll grants from this run have a scope_curated decision - review step complete, continuing.")
        mark_done(status, 'review', STATUS_DIR)
    else:
        print(f"\nStep 4 (review) requires a manual step:"
              f"\n  1. Run S5_Review_classifications.ipynb - it auto-curates confident grants and exports"
              f"\n     the ambiguous ones (from this run and any other pending grants) for review."
              f"\n  2. Review the exported CSV, then complete the notebook's second section to apply your"
              f"\n     decisions back onto funding_classified."
              f"\n  3. Re-run this script to continue (it will pick up from here).")
        sys.exit(0)
else:
    print("\nStep 4 (review) already done, skipping.")

# Step 5: Research category labelling + promotion into funding_curated
if status['steps']['llm_labelling'] != 'done':
    print("\nStarting Step 5: Research category labelling.")
    label_research_category(
        KEY_PATH=cfg['KEY_PATH'],
        DB_PATH=cfg['DB_PATH'],
        CLASSIFICATION_TABLE=cfg['CLASSIFICATION_TABLE'],
        LLM_MODEL_LABEL=cfg['LLM_MODEL_LABEL'],
    )
    mark_done(status, 'llm_labelling', STATUS_DIR)
else:
    print("\nStep 5 (llm_labelling) already done, skipping.")

print(f"\nPipeline complete for run '{cfg['RUN_TABLE']}'.")
