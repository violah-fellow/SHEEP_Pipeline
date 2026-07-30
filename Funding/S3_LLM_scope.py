## LLM scoping script: send deduplicated new grants to Claude via the Batch API and store scope/pillar results.
## Submits a batch, waits (polling) for it to complete, then parses and writes results back to the database.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().
## Unlike Publications, there is no ML gate here - every row in DEDUP_TABLE is sent to the LLM.

import os

# CONFIG
# edit parameters for this run here

# Anthropic API
# path to API key
KEY_PATH = '../.env'

# Database
# path to DuckDB database (self-contained in Pipeline/Funding, mirrors Publications)
DB_PATH = 'funding.db'
# table containing the raw run data (only used for batch metadata file naming)
RUN_TABLE = 'dim_query_test'
# table containing the deduplicated new grants to classify - produced by S2_grant_deduplication.ipynb
DEDUP_TABLE = 'dim_query_test_dedup'
# table for final classifications
CLASSIFICATION_TABLE = 'funding_classified'

# LLM
# path to the system prompt used for scoping
PROMPT_PATH = 'llm_prompts/scope_prompt_funding.md'
# model to use for scoping; set from the main pipeline script
LLM_MODEL_SCOPE = 'claude-sonnet-4-6'  # or 'claude-sonnet-4-6' for more accurate results
MAX_TOKENS = 512
TEMPERATURE = 0.0

# Batch
# directory for batch submission metadata, keyed by RUN_TABLE (allows resuming without resubmitting)
BATCH_DIR = 'batch_jobs'
# max requests per batch submission - a single submission of thousands of requests (tens of MB
# of JSON, since the system prompt and tool definition are repeated per request) is prone to
# gateway/proxy timeouts, so large runs are split into chunks of this size and submitted as
# separate batches
BATCH_CHUNK_SIZE = 500
# how often to check whether the batch has finished
POLL_INTERVAL_SECONDS = 600

# START OF SCRIPT

_TOOL_PROPERTIES = {
    "scope": {
        "type": "string",
        "enum": ["in", "out"],
        "description": "Whether the grant is in scope for alternative proteins."
    },
    "confidence": {
        "type": "integer",
        "minimum": 1,
        "maximum": 7,
        "description": "Confidence score 1-7 for the scope decision."
    },
    "plant_based": {"type": "boolean"},
    "fermentation": {"type": "boolean"},
    "cultivated": {"type": "boolean"},
    "cross_cutting": {"type": "boolean"},
}
_TOOL_REQUIRED = ["scope", "confidence", "plant_based", "fermentation", "cultivated", "cross_cutting"]

CLASSIFICATION_TOOL = {
    "name": "classify_grant",
    "description": "Record the scope and pillar classification for a funding grant.",
    "input_schema": {
        "type": "object",
        "properties": _TOOL_PROPERTIES,
        "required": _TOOL_REQUIRED,
    }
}


def main(
    KEY_PATH=KEY_PATH,
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    DEDUP_TABLE=DEDUP_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    PROMPT_PATH=PROMPT_PATH,
    LLM_MODEL_SCOPE=LLM_MODEL_SCOPE,
    MAX_TOKENS=MAX_TOKENS,
    TEMPERATURE=TEMPERATURE,
    BATCH_DIR=BATCH_DIR,
    BATCH_CHUNK_SIZE=BATCH_CHUNK_SIZE,
    POLL_INTERVAL_SECONDS=POLL_INTERVAL_SECONDS,
):
    import time
    import json
    from datetime import datetime
    from pathlib import Path

    import anthropic
    import duckdb
    import pandas as pd
    from dotenv import load_dotenv

    # 1. Authenticate with the Anthropic API
    print("\nConnecting to the Anthropic API")

    load_dotenv(KEY_PATH)
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    batch_dir = Path(BATCH_DIR)
    batch_dir.mkdir(exist_ok=True)
    metadata_path = batch_dir / f"{RUN_TABLE}_llm_scope.json"

    db = duckdb.connect(database=DB_PATH)

    # 2. Ensure CLASSIFICATION_TABLE exists and contains DEDUP_TABLE's rows.
    # Unlike Publications (where S1 already populates RUN_TABLE, which CLASSIFICATION_TABLE
    # shares), funding_classified only ever receives rows that survived deduplication - so it
    # needs to be seeded/grown from DEDUP_TABLE here, before batch submission. Idempotent, so
    # safe to run on every invocation (including resumes).
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE not in existing_tables:
        db.sql(f"CREATE TABLE {CLASSIFICATION_TABLE} AS SELECT * FROM {DEDUP_TABLE} LIMIT 0")
        print(f"Created empty '{CLASSIFICATION_TABLE}' with {DEDUP_TABLE}'s schema.")
    n_before = db.sql(f"SELECT COUNT(*) AS n FROM {CLASSIFICATION_TABLE}").df()['n'][0]
    db.sql(f"""
        INSERT INTO {CLASSIFICATION_TABLE}
        SELECT * FROM {DEDUP_TABLE}
        WHERE "Grant ID" NOT IN (SELECT "Grant ID" FROM {CLASSIFICATION_TABLE})
    """)
    n_after = db.sql(f"SELECT COUNT(*) AS n FROM {CLASSIFICATION_TABLE}").df()['n'][0]
    print(f"Inserted {n_after - n_before} new rows from '{DEDUP_TABLE}' into '{CLASSIFICATION_TABLE}'.")

    # 3. Load input data
    if metadata_path.exists():
        # Batch(es) for this run were already submitted; resume from their metadata instead of resubmitting.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        batch_ids = [b["batch_id"] for b in metadata["batches"]]
        print(f"Found {len(batch_ids)} existing batch(es) for '{RUN_TABLE}': {batch_ids}. Resuming without resubmitting.")
    else:
        data = db.sql(f"SELECT * FROM {DEDUP_TABLE}").df()
        print(f"{len(data)} rows loaded from '{DEDUP_TABLE}'")

        if len(data) == 0:
            print("No rows to submit. Skipping batch.")
            db.close()
            return

        # 4. Build and submit batch requests
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        def build_batch_request(row):
            user_message = f"Title: {row['Title translated']}\n\nAbstract: {row['Abstract translated']}"
            return {
                "custom_id": row["Grant ID"].replace(".", "_"),
                "params": {
                    "model": LLM_MODEL_SCOPE,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [{"role": "user", "content": user_message}],
                    "tools": [CLASSIFICATION_TOOL],
                    "tool_choice": {"type": "tool", "name": "classify_grant"},
                }
            }

        batch_requests = [build_batch_request(row) for _, row in data.iterrows()]
        grant_ids = data["Grant ID"].tolist()

        print(f"\nSubmitting {len(batch_requests)} requests in chunks of {BATCH_CHUNK_SIZE}")
        batches_metadata = []
        for start in range(0, len(batch_requests), BATCH_CHUNK_SIZE):
            chunk = batch_requests[start:start + BATCH_CHUNK_SIZE]
            chunk_ids = grant_ids[start:start + BATCH_CHUNK_SIZE]
            batch = client.messages.batches.create(requests=chunk)
            print(f"Batch {len(batches_metadata) + 1}: {batch.id} ({len(chunk)} requests, status: {batch.processing_status})")
            batches_metadata.append({
                "batch_id": batch.id,
                "n_records": len(chunk),
                "dataset_ids": chunk_ids,
            })

        batch_ids = [b["batch_id"] for b in batches_metadata]

        metadata = {
            "run_table": RUN_TABLE,
            "dedup_table": DEDUP_TABLE,
            "model": LLM_MODEL_SCOPE,
            "n_records": len(data),
            "batches": batches_metadata,
            "created_at": datetime.now().isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Batch metadata saved to {metadata_path}")

    # 5. Poll until all batches have finished
    print(f"\nWaiting for {len(batch_ids)} batch(es) to complete")

    pending = set(batch_ids)
    while pending:
        for batch_id in list(pending):
            batch = client.messages.batches.retrieve(batch_id)
            print(f"Batch {batch_id}: {batch.processing_status}   Counts: {batch.request_counts}")
            if batch.processing_status == "ended":
                pending.discard(batch_id)
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)

    # 6. Retrieve and parse results from every batch
    print("\nRetrieving results")

    raw_results = []
    for batch_id in batch_ids:
        for result in client.messages.batches.results(batch_id):
            grant_id = result.custom_id.replace("_", ".")
            if result.result.type == "succeeded":
                content = result.result.message.content
                stop_reason = result.result.message.stop_reason
                tool_block = next((b for b in content if b.type == "tool_use"), None)
                if tool_block:
                    record = dict(tool_block.input)
                    record["Grant ID"] = grant_id
                    record["status_LLM"] = "ok"
                    record["stop_reason_LLM"] = stop_reason
                else:
                    record = {"Grant ID": grant_id, "status_LLM": "parse_error", "stop_reason_LLM": stop_reason}
            else:
                record = {"Grant ID": grant_id, "status_LLM": result.result.type, "stop_reason_LLM": None}
            raw_results.append(record)

    results_df = pd.DataFrame(raw_results)
    non_llm_cols = {"Grant ID", "status_LLM", "stop_reason_LLM"}
    results_df = results_df.rename(columns={c: f"{c}_LLM" for c in results_df.columns if c not in non_llm_cols})

    n_ok = (results_df["status_LLM"] == "ok").sum()
    n_missing_scope = results_df["scope_LLM"].isna().sum() if "scope_LLM" in results_df.columns else len(results_df)
    print(f"Results: {len(results_df)} total, {n_ok} succeeded, {n_missing_scope} missing a scope decision.")
    if "scope_LLM" in results_df.columns:
        print(f"LLM predicted in scope: {(results_df['scope_LLM'] == 'in').sum()}")

    # Derive pillar_LLM from the boolean flags (same logic as Publications' S3_LLM_scope.py):
    # CC if multiple pillar flags are True, or if only cross_cutting_LLM is True
    pillar_flags = ["plant_based_LLM", "fermentation_LLM", "cultivated_LLM"]

    def derive_pillar(row):
        if row["status_LLM"] != "ok":
            return None
        n_flags = sum(bool(row[f]) for f in pillar_flags)
        if n_flags > 1 or (row["cross_cutting_LLM"] and n_flags == 0):
            return "CC"
        if row["plant_based_LLM"]:
            return "PB"
        if row["fermentation_LLM"]:
            return "F"
        if row["cultivated_LLM"]:
            return "CM"
        return "NA"

    results_df["pillar_LLM"] = results_df.apply(derive_pillar, axis=1)
    date_LLM = datetime.today().strftime('%y%m%d')
    results_df["date_LLM"] = date_LLM

    # store retrieval date in batch metadata so the log reflects when results were collected
    metadata["date_LLM"] = date_LLM
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 7. Write results back to DEDUP_TABLE and CLASSIFICATION_TABLE
    print(f"\nUpdating '{DEDUP_TABLE}' and '{CLASSIFICATION_TABLE}' with LLM columns.")

    llm_columns = {
        'scope_LLM':         'VARCHAR',
        'confidence_LLM':    'DOUBLE',
        'pillar_LLM':        'VARCHAR',
        'plant_based_LLM':   'BOOLEAN',
        'fermentation_LLM':  'BOOLEAN',
        'cultivated_LLM':    'BOOLEAN',
        'cross_cutting_LLM': 'BOOLEAN',
        'status_LLM':        'VARCHAR',
        'stop_reason_LLM':   'VARCHAR',
        'date_LLM':          'VARCHAR',
    }
    results_df = results_df.reindex(columns=['Grant ID'] + list(llm_columns.keys()))

    for table in (DEDUP_TABLE, CLASSIFICATION_TABLE):
        existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
        if table not in existing_tables:
            print(f"'{table}' does not exist yet, skipping.")
            continue

        for col, dtype in llm_columns.items():
            db.sql(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS {col} {dtype}")

        db.register('llm_results', results_df)
        set_clause = ", ".join(f"{col} = llm_results.{col}" for col in llm_columns)
        db.sql(f"""
            UPDATE {table}
            SET {set_clause}
            FROM llm_results
            WHERE {table}."Grant ID" = llm_results."Grant ID"
        """)
        print(f"'{table}' updated with LLM columns for {len(results_df)} rows.")

    db.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
