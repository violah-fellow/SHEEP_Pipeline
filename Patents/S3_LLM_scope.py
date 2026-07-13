## LLM scoping script: send ML-positive patents to Claude via the Batch API and store scope/pillar results.
## Submits a batch, waits (polling) for it to complete, then parses and writes results back to the database.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import os

# CONFIG
# edit parameters for this run here

# Anthropic API
# path to API key
KEY_PATH = '../.env'

# Database
# path to DuckDB database
DB_PATH = 'patents.db'
# table containing the new input data to classify with run date as name
RUN_TABLE = 'data_run_test'
# table for final classifications
CLASSIFICATION_TABLE = 'patents_classified'

# LLM
# path to the system prompt used for scoping
PROMPT_PATH = 'llm_prompts/scope_prompt_patents.md'
# model to use for scoping; set from the main pipeline script
LLM_MODEL_SCOPE = 'claude-haiku-4-5'  # or 'claude-sonnet-4-6' for more accurate results
MAX_TOKENS = 512
TEMPERATURE = 0.0

# Batch
# directory for batch submission metadata, keyed by RUN_TABLE (allows resuming without resubmitting)
BATCH_DIR = 'batch_jobs'
# how often to check whether the batch has finished
POLL_INTERVAL_SECONDS = 600

# START OF SCRIPT

_TOOL_PROPERTIES = {
    "scope": {
        "type": "string",
        "enum": ["in", "out"],
        "description": "Whether the patent is in scope for alternative proteins."
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
    "name": "classify_patent",
    "description": "Record the scope and pillar classification for a patent.",
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
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    PROMPT_PATH=PROMPT_PATH,
    LLM_MODEL_SCOPE=LLM_MODEL_SCOPE,
    MAX_TOKENS=MAX_TOKENS,
    TEMPERATURE=TEMPERATURE,
    BATCH_DIR=BATCH_DIR,
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

    # 2. Load and filter input data
    db = duckdb.connect(database=DB_PATH)

    if metadata_path.exists():
        # A batch for this run was already submitted; resume from its metadata instead of resubmitting.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        batch_id = metadata["batch_id"]
        print(f"Found existing batch for '{RUN_TABLE}': {batch_id}. Resuming without resubmitting.")
    else:
        data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()
        print(f"{len(data)} rows loaded from '{RUN_TABLE}'")

        data = data[data['pred_combined'] == 1].reset_index(drop=True)
        print(f"{len(data)} rows flagged as in scope by ML, sending to LLM.")

        if len(data) == 0:
            print("No rows to submit. Skipping batch.")
            db.close()
            return

        # 3. Build and submit batch requests
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        def build_batch_request(row):
            user_message = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
            return {
                "custom_id": row["id"].replace(".", "_"),
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
                    "tool_choice": {"type": "tool", "name": "classify_patent"},
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
            "run_table": RUN_TABLE,
            "model": LLM_MODEL_SCOPE,
            "n_records": len(data),
            "dataset_ids": data["id"].tolist(),
            "created_at": datetime.now().isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Batch metadata saved to {metadata_path}")

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
        patent_id = result.custom_id.replace("_", ".")
        if result.result.type == "succeeded":
            content = result.result.message.content
            stop_reason = result.result.message.stop_reason
            tool_block = next((b for b in content if b.type == "tool_use"), None)
            if tool_block:
                record = dict(tool_block.input)
                record["id"] = patent_id
                record["status_LLM"] = "ok"
                record["stop_reason_LLM"] = stop_reason
            else:
                record = {"id": patent_id, "status_LLM": "parse_error", "stop_reason_LLM": stop_reason}
        else:
            record = {"id": patent_id, "status_LLM": result.result.type, "stop_reason_LLM": None}
        raw_results.append(record)

    results_df = pd.DataFrame(raw_results)
    non_llm_cols = {"id", "status_LLM", "stop_reason_LLM"}
    results_df = results_df.rename(columns={c: f"{c}_LLM" for c in results_df.columns if c not in non_llm_cols})

    n_ok = (results_df["status_LLM"] == "ok").sum()
    n_missing_scope = results_df["scope_LLM"].isna().sum() if "scope_LLM" in results_df.columns else len(results_df)
    print(f"Results: {len(results_df)} total, {n_ok} succeeded, {n_missing_scope} missing a scope decision.")
    if "scope_LLM" in results_df.columns:
        print(f"LLM predicted in scope: {(results_df['scope_LLM'] == 'in').sum()}")

    # Derive pillar_LLM from the boolean flags:
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

    # 6. Write results back to RUN_TABLE and CLASSIFICATION_TABLE
    print(f"\nUpdating '{RUN_TABLE}' with LLM columns.")

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
    results_df = results_df.reindex(columns=['id'] + list(llm_columns.keys()))

    # NaN/non-bool values in these columns cause DuckDB to fail casting to BOOL;
    # convert explicitly to pandas nullable boolean (NaN/unknown → pd.NA → NULL)
    def _safe_bool(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return pd.NA
        if isinstance(x, bool):
            return x
        if isinstance(x, str):
            return x.lower() not in ('false', '0', 'no', '')
        return bool(x)

    for col in ('plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM'):
        if col in results_df.columns:
            results_df[col] = pd.array([_safe_bool(x) for x in results_df[col]], dtype='boolean')

    for table in (RUN_TABLE, CLASSIFICATION_TABLE):
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
            WHERE {table}.id = llm_results.id
        """)
        print(f"'{table}' updated with LLM columns for {len(results_df)} rows.")

    db.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
