## STEP 6 of the publications pipeline - the last step before the export (S7)
## LLM research-category labelling script: sends in-scope, pillar-curated publications to Claude
## via the Batch API and stores the resulting research_category in publications_labelled.
##
## What it does: while S3 only decided in/out of scope and pillar, this step assigns the detailed
## research category (e.g. "Ingredient optimisation", "Bioprocess design"). The available categories
## differ per pillar, so there is one system prompt and one tool schema per pillar - each publication
## is asked with the category list of ITS pillar. Everything goes into a single batch nevertheless.
##
## Prerequisite: the manual review (S5_Review_classifications.ipynb) must be finished, because this
## step reads scope_curated / pillar_curated - the reviewed columns, not the LLM's own guess.
##
## Input:   CLASSIFICATION_TABLE, rows with scope_curated == 'in' and a valid pillar_curated
## Output:  diagnostic *_LLM columns in CLASSIFICATION_TABLE, plus LABELLED_TABLE - the clean final
##          table (Dimensions metadata + curated scope/pillar + research_category) that
##          S7_Retrieve_dataset.ipynb exports
##
## Only successfully labelled rows are written to LABELLED_TABLE. Failed ones simply stay missing
## from it, which means the next run picks them up again automatically - running this script twice is
## safe and is the normal way to retry failures.
## Standalone for now (not called from pipeline_publications.py)
## wrapped in main() so it can be wired into the pipeline later without changes.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import os

# CONFIG
# edit parameters for this run here

# Anthropic API
# path to API key
KEY_PATH = '../.env'

# Database
# path to DuckDB database
DB_PATH = 'publications.db'
# table with in-scope, pillar-classified publications to label (working table; gets LLM
# diagnostic columns added to it, same pattern as S3_LLM_scope.py)
CLASSIFICATION_TABLE = 'publications_classified'
# table for the final clean labelled output
LABELLED_TABLE = 'publications_labelled'

# Curated columns
# these are added to CLASSIFICATION_TABLE by manual review and don't exist yet as of writing
# this script; update the names here if they end up called something else
# (both columns exist by now - they are created by S5_Review_classifications.ipynb. If this script
# fails with "column not found", the review step has not been run for this data yet.)
SCOPE_COL = 'scope_curated'
PILLAR_COL = 'pillar_curated'

# LLM
# one system prompt per pillar; files are added separately to llm_prompts/
# each prompt describes the categories of that pillar; the enum of allowed values is built from
# PILLAR_CATS further down. Prompt and category list must stay consistent - if you add a category to
# a prompt file, add it to the corresponding *_CATS list as well, otherwise the model cannot return it.
PROMPT_PATHS = {
    'PB': 'llm_prompts/category_prompt_publications_PB.md',
    'F':  'llm_prompts/category_prompt_publications_F.md',
    'CM': 'llm_prompts/category_prompt_publications_CM.md',
    'CC': 'llm_prompts/category_prompt_publications_CC.md',
}
LLM_MODEL_LABEL = 'claude-sonnet-4-6'  # or 'claude-haiku-4-5' for cheap test runs
# the answer is only two category names, so a small limit is enough here (S3 allows 512)
MAX_TOKENS = 150
# 0.0 = deterministic, as in S3: category assignment should be reproducible
TEMPERATURE = 0.0

# Batch
# directory for batch submission metadata, keyed by RUN_LABEL (allows resuming without resubmitting)
BATCH_DIR = 'batch_jobs'
# how often to check whether the batch has finished
# 300 s = 5 min; shorter than in S3 because these requests are small and usually finish quickly
POLL_INTERVAL_SECONDS = 300
# identifies this run's batch metadata file. Leave as None: on each call, the script first looks
# in BATCH_DIR for a run that was submitted but never finished (no "completed_at" in its metadata,
# same idea as find_incomplete_run() in Helper_pipeline_functions.py) and resumes that; only if
# none exists does it mint a fresh timestamped label and submit a new batch. Pass a value explicitly
# to force resuming (or starting) a specific run.
RUN_LABEL = None

# Research categories allowed per pillar, used to build each tool call's enum.
# Hardcoded from the latest tested prompt version (see Publications/GenAI/genai_rescat_testing.ipynb)
# update by hand if the categories in the prompt files change.
# the lists differ per pillar on purpose (e.g. "Cell line development" only exists for CM), and
# several categories appear in more than one pillar. The order does not matter; the exact spelling
# does, because the strings the model returns are matched against these lists (see
# _normalise_category() - only the letter case is tolerated).
PB_CATS = ["Crop development", "Strain development", "Ingredient optimisation", "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition", "Consumer & market research", "Impact assessments", "Other"]
F_CATS  = ["Feedstocks", "Target molecule selection", "Strain development", "Bioprocess design", "Ingredient optimisation", "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition", "Consumer & market research", "Impact assessments", "Other"]
CM_CATS = ["Cell line development", "Cell culture media", "Bioprocess design", "Scaffolding", "End product formulation", "Food safety & quality", "Health & nutrition", "Consumer & market research", "Impact assessments", "Other"]
CC_CATS = ["Bioprocess design", "Scaffolding", "Ingredient optimisation", "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition", "Consumer & market research", "Impact assessments", "Other"]
PILLAR_CATS = {'PB': PB_CATS, 'F': F_CATS, 'CM': CM_CATS, 'CC': CC_CATS}

# columns on CLASSIFICATION_TABLE that are ML/LLM working columns, not Dimensions metadata;
# excluded when building LABELLED_TABLE rows
NON_METADATA_COLS = {
    'pred_combined', 'pred_pillar',
    'scope_LLM', 'confidence_LLM', 'pillar_LLM',
    'plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
    'status_LLM', 'stop_reason_LLM',
    'primary_category_LLM', 'secondary_category_LLM',
    'category_status_LLM', 'category_stop_reason_LLM',
}
# nested Dimensions fields that need JSON serialization before being written to a new table:
# their native shape (STRUCT vs MAP) can differ between queries, which crashes a plain CREATE
# TABLE / INSERT (see S2_ML_classification.py)
NESTED_JSON_COLS = ('authors', 'funder_countries', 'research_org_cities', 'research_org_countries')

# START OF SCRIPT

## _build_category_tool
## builds the tool definition for one pillar: the 'enum' restricts the answer to that pillar's
## categories, and both 'primary' and 'secondary' are required so a second-best category is always
## returned as well (stored for QA, but only 'primary' ends up in the final dataset)
def _build_category_tool(categories):
    return {
        "name": "label_research_category",
        "description": "Record the primary and secondary research category for a research publication.",
        "input_schema": {
            "type": "object",
            "properties": {
                "primary": {
                    "type": "string",
                    "enum": categories,
                    "description": "The research category that best captures the paper's primary focus."
                },
                "secondary": {
                    "type": "string",
                    "enum": categories,
                    "description": "The second most relevant research category."
                },
            },
            "required": ["primary", "secondary"],
        }
    }


## _normalise_category
## maps a category string returned by the model back onto the official spelling
def _normalise_category(value, categories):
    # the LLM sometimes returns different casing (e.g. "Health & Nutrition") despite the enum
    # constraint; look up case-insensitively and treat anything else as unparseable
    # anything that is not in the list becomes None, which makes the row a 'parse_error' below -
    # i.e. invented categories never reach the database
    if not isinstance(value, str):
        return None
    return {c.lower(): c for c in categories}.get(value.strip().lower())


def main(
    KEY_PATH=KEY_PATH,
    DB_PATH=DB_PATH,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    LABELLED_TABLE=LABELLED_TABLE,
    SCOPE_COL=SCOPE_COL,
    PILLAR_COL=PILLAR_COL,
    PROMPT_PATHS=PROMPT_PATHS,
    PILLAR_CATS=PILLAR_CATS,
    LLM_MODEL_LABEL=LLM_MODEL_LABEL,
    MAX_TOKENS=MAX_TOKENS,
    TEMPERATURE=TEMPERATURE,
    BATCH_DIR=BATCH_DIR,
    POLL_INTERVAL_SECONDS=POLL_INTERVAL_SECONDS,
    RUN_LABEL=RUN_LABEL,
):
    import time
    import json
    from datetime import datetime
    from pathlib import Path

    import anthropic
    import duckdb
    import numpy as np
    import pandas as pd
    from dotenv import load_dotenv

    # 1. Authenticate with the Anthropic API
    print("\nConnecting to the Anthropic API")

    load_dotenv(KEY_PATH)
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    batch_dir = Path(BATCH_DIR)
    batch_dir.mkdir(exist_ok=True)

    def find_incomplete_batch():
        """Return the run_label of the most recent labelling batch that was submitted but never
        finished writing results (no 'completed_at' in its metadata), or None if there isn't one."""
        files = sorted(
            (f for f in os.listdir(batch_dir) if f.endswith('_llm_labelling.json')),
            reverse=True,
        )
        for fname in files:
            meta = json.loads((batch_dir / fname).read_text(encoding="utf-8"))
            if 'completed_at' not in meta:
                return meta['run_label']
        return None
    
    def mark_batch_complete():
        # marks this run so find_incomplete_batch() skips it on future calls
        metadata['completed_at'] = datetime.now().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if RUN_LABEL is None:
        RUN_LABEL = find_incomplete_batch()
    if RUN_LABEL is None:
        RUN_LABEL = f"labelling_{datetime.today().strftime('%y%m%d_%H%M')}"
        print(f"Starting new run '{RUN_LABEL}'")
    else:
        print(f"Resuming run '{RUN_LABEL}'")

    metadata_path = batch_dir / f"{RUN_LABEL}_llm_labelling.json"

    # 2. Load and filter input data. The connection is only open for this lookup, not for the
    # submission/poll/parse steps below, so it isn't held locked for however long the batch takes.
    if metadata_path.exists():
        # A batch for this run was already submitted; resume from its metadata instead of resubmitting.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        batch_id = metadata["batch_id"]
        print(f"Found existing batch for '{RUN_LABEL}': {batch_id}. Resuming without resubmitting.")
    else:
        with duckdb.connect(database=DB_PATH) as db:
            data = db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE}").df()
            print(f"{len(data)} rows loaded from '{CLASSIFICATION_TABLE}'")

            # the curated scope, not scope_LLM: only what a human confirmed (or what the review
            # notebook accepted automatically) is labelled here
            data = data[data[SCOPE_COL] == 'in'].reset_index(drop=True)
            print(f"{len(data)} rows in scope (curated).")

            existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
            if LABELLED_TABLE in existing_tables:
                labelled_ids = db.sql(f"SELECT id FROM {LABELLED_TABLE}").df()['id']
                n_before = len(data)
                data = data[~data['id'].isin(labelled_ids)].reset_index(drop=True)
                print(f"{n_before - len(data)} rows already in '{LABELLED_TABLE}', skipped.")
            else:
                print(f"'{LABELLED_TABLE}' does not exist yet, will be created.")

            # rows without a usable pillar are dropped: there is no category list for them, so they
            # cannot be labelled. If this number is unexpectedly high, the review step probably left
            # pillar_curated empty (or set to 'manual_review') for those rows.
            n_before = len(data)
            data = data[data[PILLAR_COL].isin(PILLAR_CATS.keys())].reset_index(drop=True)
            if n_before - len(data) > 0:
                print(f"{n_before - len(data)} rows dropped for missing/unrecognised '{PILLAR_COL}'.")

        if len(data) == 0:
            print("No new rows to label.")
            return

        # 3. Build and submit batch requests: one system prompt/tool schema per pillar, all
        # requests combined into a single batch submission
        system_prompts = {}
        tools = {}
        for pillar, prompt_path in PROMPT_PATHS.items():
            with open(prompt_path, "r", encoding="utf-8") as f:
                system_prompts[pillar] = f.read().strip()
            tools[pillar] = _build_category_tool(PILLAR_CATS[pillar])

        # the pillar decides which prompt and which tool schema a request gets - that is the only
        # difference between the requests in this batch
        def build_batch_request(row):
            pillar = row[PILLAR_COL]
            user_message = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
            return {
                "custom_id": row["id"].replace(".", "_"),
                "params": {
                    "model": LLM_MODEL_LABEL,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompts[pillar],
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [{"role": "user", "content": user_message}],
                    "tools": [tools[pillar]],
                    "tool_choice": {"type": "tool", "name": "label_research_category"},
                }
            }

        batch_requests = [build_batch_request(row) for _, row in data.iterrows()]

        print("\nSubmitting batch")
        batch = client.messages.batches.create(requests=batch_requests)
        batch_id = batch.id
        print(f"Batch ID: {batch_id}")
        print(f"Status:   {batch.processing_status}")

        metadata = {
            "batch_id": batch_id,
            "run_label": RUN_LABEL,
            "model": LLM_MODEL_LABEL,
            "n_records": len(data),
            "dataset_ids": data["id"].tolist(),
            # persisted so the resume path can still normalise categories per pillar
            "pillar_by_id": dict(zip(data["id"], data[PILLAR_COL])),
            "created_at": datetime.now().isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Batch metadata saved to {metadata_path}")

    # needed when parsing the results: to check a returned category, you have to know which pillar's
    # list it belongs to. Read from the metadata file rather than from the database, so a resumed run
    # uses the pillar assignment as it was at submission time.
    pillar_by_id = metadata["pillar_by_id"]

    # 4. Poll until the batch has finished
    print("\nWaiting for batch to complete")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        print(f"Processing status: {batch.processing_status}   Counts: {batch.request_counts}")
        if batch.processing_status == "ended":
            break
        time.sleep(POLL_INTERVAL_SECONDS)

    # 5. Retrieve and parse results
    print("\nRetrieving results")

    raw_results = []
    for result in client.messages.batches.results(batch_id):
        pub_id = result.custom_id.replace("_", ".")
        categories = PILLAR_CATS.get(pillar_by_id.get(pub_id), [])
        if result.result.type == "succeeded":
            content = result.result.message.content
            stop_reason = result.result.message.stop_reason
            tool_block = next((b for b in content if b.type == "tool_use"), None)
            primary = _normalise_category(tool_block.input.get("primary"), categories) if tool_block else None
            if primary is None:
                record = {"id": pub_id, "status": "parse_error", "stop_reason": stop_reason}
            else:
                record = {
                    "id": pub_id,
                    "primary": primary,
                    "secondary": _normalise_category(tool_block.input.get("secondary"), categories),
                    "status": "ok",
                    "stop_reason": stop_reason,
                }
        else:
            record = {"id": pub_id, "status": result.result.type, "stop_reason": None}
        raw_results.append(record)

    results_df = pd.DataFrame(raw_results)
    results_df = results_df.rename(columns={
        "primary":     "primary_category_LLM",
        "secondary":   "secondary_category_LLM",
        "status":      "category_status_LLM",
        "stop_reason": "category_stop_reason_LLM",
    })

    # worth checking: only the 'ok' rows end up in LABELLED_TABLE. A large gap between the two
    # numbers means many requests failed or returned an unusable category - those rows are picked up
    # again on the next run.
    n_ok = (results_df["category_status_LLM"] == "ok").sum()
    print(f"Results: {len(results_df)} total, {n_ok} succeeded.")

    date_labelling = datetime.today().strftime('%y%m%d')
    results_df["date_labelling"] = date_labelling

    # store retrieval date in batch metadata so the log reflects when results were collected
    metadata["date_labelling"] = date_labelling
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 6. Write diagnostic LLM columns (primary + secondary category, status, stop reason) back to
    # CLASSIFICATION_TABLE for QA; date_labelling is also written here so it flows to LABELLED_TABLE.
    print(f"\nUpdating '{CLASSIFICATION_TABLE}' with LLM category columns.")

    llm_columns = {
        'primary_category_LLM':     'VARCHAR',
        'secondary_category_LLM':   'VARCHAR',
        'category_status_LLM':      'VARCHAR',
        'category_stop_reason_LLM': 'VARCHAR',
        'date_labelling':           'VARCHAR',
    }
    results_df = results_df.reindex(columns=['id'] + list(llm_columns.keys()))

    with duckdb.connect(database=DB_PATH) as db:
        for col, dtype in llm_columns.items():
            db.sql(f"ALTER TABLE {CLASSIFICATION_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype}")

        db.register('llm_results', results_df)
        set_clause = ", ".join(f"{col} = llm_results.{col}" for col in llm_columns)
        db.sql(f"""
            UPDATE {CLASSIFICATION_TABLE}
            SET {set_clause}
            FROM llm_results
            WHERE {CLASSIFICATION_TABLE}.id = llm_results.id
        """)
        print(f"'{CLASSIFICATION_TABLE}' updated with LLM category columns for {len(results_df)} rows.")

    # 7. Build clean rows for LABELLED_TABLE: Dimensions metadata columns + curated scope/pillar
    # + research_category. Rows that failed or didn't parse are left out so they get retried on
    # the next run (they'll still be missing from LABELLED_TABLE).
    ok_ids = results_df.loc[results_df["category_status_LLM"] == "ok", "id"].tolist()

    with duckdb.connect(database=DB_PATH) as db:
        # guard against duplicate inserts if this run is resumed after already writing results once
        # (e.g. a crash between the LABELLED_TABLE write below and mark_batch_complete())
        existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
        if LABELLED_TABLE in existing_tables:
            already_labelled = set(db.sql(f"SELECT id FROM {LABELLED_TABLE}").df()['id'])
            n_before = len(ok_ids)
            ok_ids = [i for i in ok_ids if i not in already_labelled]
            if n_before - len(ok_ids) > 0:
                print(f"{n_before - len(ok_ids)} rows already in '{LABELLED_TABLE}' (resumed run), skipped.")

        if not ok_ids:
            print("\nNo successfully labelled rows to add to LABELLED_TABLE.")
            mark_batch_complete()
            print("\nDone!")
            return

        all_columns = db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
        dimensions_columns = [c for c in all_columns if c not in NON_METADATA_COLS]

        db.register('ok_ids', pd.DataFrame({'id': ok_ids}))
        labelled_data = db.sql(f"""
            SELECT {', '.join(f'"{c}"' for c in dimensions_columns)}
            FROM {CLASSIFICATION_TABLE}
            JOIN ok_ids USING (id)
        """).df()

        # the final table only carries the primary category, under the plain name 'research_category'.
        # secondary_category_LLM stays in CLASSIFICATION_TABLE for QA and is not exported.
        labelled_data = labelled_data.merge(
            results_df.loc[results_df["category_status_LLM"] == "ok", ["id", "primary_category_LLM"]],
            on="id", how="left"
        ).rename(columns={"primary_category_LLM": "research_category"})

        # serialize nested Dimensions fields to JSON text (see S2_ML_classification.py for why)
        def _to_json(x):
            if x is None:
                return None
            if isinstance(x, np.ndarray):
                x = x.tolist()
            return json.dumps(x, default=str)

        for col in NESTED_JSON_COLS:
            if col in labelled_data.columns:
                labelled_data[col] = labelled_data[col].apply(_to_json)

        print(f"\nAdding {len(labelled_data)} rows to '{LABELLED_TABLE}'.")

        db.register('labelled_data', labelled_data)
        existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
        if LABELLED_TABLE in existing_tables:
            cols_sql = ", ".join(f'"{c}"' for c in labelled_data.columns)
            db.sql(f"INSERT INTO {LABELLED_TABLE} ({cols_sql}) SELECT * FROM labelled_data")
        else:
            db.sql(f"CREATE TABLE {LABELLED_TABLE} AS SELECT * FROM labelled_data")
        print(f"'{LABELLED_TABLE}' now contains the newly labelled rows.")

        mark_batch_complete()

    print("\nDone!")


# only executed when the file is started directly (python S6_LLM_labelling.py) - this step is not
# part of pipeline_publications.py and is currently always started by hand after the review
if __name__ == '__main__':
    main()
