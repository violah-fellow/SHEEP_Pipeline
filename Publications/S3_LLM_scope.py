## STEP 3 of the publications pipeline
## LLM scoping script: send ML-positive publications to Claude via the Batch API and store scope/pillar results.
## Submits a batch, waits (polling) for it to complete, then parses and writes results back to the database.
##
## What it does: the second filter stage. Every publication the ML step marked as possibly in scope
## (pred_combined == 1) is sent to Claude with title and abstract, and Claude returns a scope
## decision, a confidence score 1-7, and four pillar flags. Those results are NOT the final verdict:
## S5_Review_classifications.ipynb decides from them which records are accepted automatically and
## which ones a human has to look at.
##
## Input:   RUN_TABLE (with the ML predictions from S2) and the system prompt in PROMPT_PATH
## Output:  the *_LLM columns in RUN_TABLE and CLASSIFICATION_TABLE
##
## Why the Batch API: batch requests cost about half as much as normal calls, at the price of taking
## up to 24 hours. The script submits the batch, then polls until it is finished - it may therefore
## run for a very long time. That is fine: everything it needs to resume is stored on disk
## (batch_jobs/<RUN_TABLE>_llm_scope.json), so the script can be interrupted and restarted at any
## time; it picks the batch up again instead of paying for a second submission.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import os

# CONFIG
# edit parameters for this run here

# NOTE: started through pipeline_publications.py, these are overridden by the pipeline's own CONFIG.

# Anthropic API
# path to API key
# the .env file containing CLAUDE_API_KEY
KEY_PATH = '../.env'

# Database
# path to DuckDB database
DB_PATH = 'publications.db'
# table containing the new input data to classify with run date as name
# must already contain the ML columns from S2_ML_classification.py - the script filters on them
RUN_TABLE = 'test_llm_scope'
# table for final classifications
# the LLM columns are written to both tables: to the run table (for this run) and to the
# accumulating classification table (which is what the review notebook works on)
CLASSIFICATION_TABLE = 'publications_classified'

# LLM
# path to the system prompt used for scoping
# contains the actual scope definition - what counts as an alternative-protein publication. This is
# the file to edit when the scope changes; the code itself contains no subject-matter rules.
PROMPT_PATH = 'llm_prompts/scope_prompt_publications.md'
# model to use for scoping; set from the main pipeline script
# haiku is markedly cheaper and quicker, sonnet more accurate - use haiku for test runs and sonnet
# for a real one (pipeline_publications.py sets sonnet)
LLM_MODEL_SCOPE = 'claude-haiku-4-5'  # or 'claude-sonnet-4-6' for more accurate results
# generous upper bound for the response; the answer is a single tool call, so far fewer tokens are
# actually used - the value only caps runaway responses
MAX_TOKENS = 512
# 0.0 = as deterministic as possible: the same publication should get the same classification, and
# this is a classification task, not a creative one
TEMPERATURE = 0.0

# Batch
# directory for batch submission metadata, keyed by RUN_TABLE (allows resuming without resubmitting)
# do not delete these json files while a batch is running - they are the only record of which batch
# belongs to which run, and without them a restart resubmits everything (and pays for it again)
BATCH_DIR = 'batch_jobs'
# how often to check whether the batch has finished
# 1800 s = 30 min. Batches usually take a few hours, so polling more often gains nothing; while
# waiting, the script does nothing but sleep and can be interrupted safely.
POLL_INTERVAL_SECONDS = 1800
# The Message Batches API caps a single batch at 256MB and 100,000 requests. When a run has
# enough in-scope publications to exceed either limit (e.g. a wide YEAR_FROM/YEAR_TO span), the
# submission is split across multiple batches automatically. Kept comfortably under the hard
# caps to leave headroom for the outer JSON structure and any estimation slack.
MAX_BATCH_BYTES = 200 * 1024 * 1024
MAX_BATCH_REQUESTS = 90000

# START OF SCRIPT

# The tool definition below is how the LLM's answer is forced into a fixed shape: instead of free
# text, Claude has to "call" this tool, and the tool's input_schema defines exactly which fields it
# must return and which values are allowed (see tool_choice further down, which makes the call
# mandatory). That is what makes the answers parseable without any text post-processing.
# The four pillar flags are booleans rather than one pillar field on purpose - a publication can
# cover several pillars, and pillar_LLM is derived from the combination of flags further below.
_TOOL_PROPERTIES = {
    "scope": {
        "type": "string",
        "enum": ["in", "out"],
        "description": "Whether the publication is in scope for alternative proteins."
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
    "name": "classify_publication",
    "description": "Record the scope and pillar classification for a research publication.",
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
    MAX_BATCH_BYTES=MAX_BATCH_BYTES,
    MAX_BATCH_REQUESTS=MAX_BATCH_REQUESTS,
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

    # the two branches below are the resume logic: if a metadata file for this run already exists,
    # the batch has been submitted before and is only collected here. Only if there is none is data
    # loaded, requests are built and a new batch is submitted (and paid for).
    if metadata_path.exists():
        # A batch for this run was already submitted; resume from its metadata instead of resubmitting.
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if "batches" not in metadata:
            # backward compat with metadata written before multi-batch chunking existed
            metadata["batches"] = [{
                "batch_id": metadata["batch_id"],
                "n_records": metadata.get("n_records"),
                "dataset_ids": metadata.get("dataset_ids", []),
            }]
        batch_ids = [b["batch_id"] for b in metadata["batches"]]
        print(f"Found existing batch(es) for '{RUN_TABLE}': {batch_ids}. Resuming without resubmitting.")
    else:
        data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()
        print(f"{len(data)} rows loaded from '{RUN_TABLE}'")

        # only what the ML step let through - this is where the actual cost saving of the two-stage
        # design comes from. Note the 1 (integer): in the run table pred_combined is 0/1, in
        # CLASSIFICATION_TABLE it is 'in'/'out' (see S2_ML_classification.py).
        data = data[data['pred_combined'] == 1].reset_index(drop=True)
        print(f"{len(data)} rows in scope, sending to LLM.")

        if len(data) == 0:
            print("No rows to submit. Skipping batch.")
            db.close()
            return

        # 3. Build and submit batch requests
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            system_prompt = f.read().strip()

        # one request per publication. Only title and abstract are sent - no journal, no authors, no
        # keywords: the decision should be based on content alone and not on where it was published.
        def build_batch_request(row):
            user_message = f"Title: {row['title']}\n\nAbstract: {row['abstract']}"
            return {
                # custom_id is how a result is matched back to its publication later. Dots are not
                # allowed in the id, so 'pub.123' is sent as 'pub_123' and converted back on parsing.
                "custom_id": row["id"].replace(".", "_"),
                "params": {
                    "model": LLM_MODEL_SCOPE,
                    "max_tokens": MAX_TOKENS,
                    "temperature": TEMPERATURE,
                    "system": [
                        {
                            "type": "text",
                            "text": system_prompt,
                            # prompt caching: the long system prompt is identical in every request,
                            # so it is only billed at full price once and read from the cache
                            # afterwards. Clearly noticeable in the cost of a large batch.
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    "messages": [{"role": "user", "content": user_message}],
                    "tools": [CLASSIFICATION_TOOL],
                    # forces the tool call: the model cannot answer in prose, it has to fill in the
                    # schema. Note it does not force completeness - individual fields can still be
                    # missing, which is what status_LLM is for.
                    "tool_choice": {"type": "tool", "name": "classify_publication"},
                }
            }

        batch_requests = [build_batch_request(row) for _, row in data.iterrows()]

        # Split into multiple batches if needed: the Batches API caps a single submission at
        # 256MB and 100,000 requests. Pack requests greedily by serialized size so each chunk
        # stays under both limits.
        def _chunk_requests(requests, max_bytes, max_count):
            chunks, current, current_size = [], [], 0
            for req in requests:
                req_size = len(json.dumps(req).encode("utf-8"))
                if req_size > max_bytes:
                    print(f"Warning: request {req['custom_id']} is {req_size} bytes, "
                          f"exceeding max_bytes ({max_bytes}) on its own; submitting it alone.")
                if current and (current_size + req_size > max_bytes or len(current) >= max_count):
                    chunks.append(current)
                    current, current_size = [], 0
                current.append(req)
                current_size += req_size
            if current:
                chunks.append(current)
            return chunks

        request_chunks = _chunk_requests(batch_requests, MAX_BATCH_BYTES, MAX_BATCH_REQUESTS)
        print(f"\nSubmitting {len(batch_requests)} requests across {len(request_chunks)} batch(es)")

        batches_meta = []
        for i, chunk in enumerate(request_chunks):
            batch = client.messages.batches.create(requests=chunk)
            print(f"Batch {i + 1}/{len(request_chunks)}: {batch.id} ({len(chunk)} requests), "
                  f"status {batch.processing_status}")
            batches_meta.append({
                "batch_id": batch.id,
                "n_records": len(chunk),
                "dataset_ids": [r["custom_id"].replace("_", ".") for r in chunk],
            })

        batch_ids = [b["batch_id"] for b in batches_meta]
        metadata = {
            "run_table": RUN_TABLE,
            "model": LLM_MODEL_SCOPE,
            "n_records": len(data),
            "dataset_ids": data["id"].tolist(),
            "batches": batches_meta,
            "created_at": datetime.now().isoformat(),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"Batch metadata saved to {metadata_path}")

    # 4. Poll until all batches have finished
    # from here on the script only waits. It may sit here for hours - that is normal, and it can be
    # interrupted with Ctrl+C: on the next start it lands in the resume branch of step 2 and
    # continues waiting for the same batch. 'ended' also covers batches that ended with errors;
    # individual failed requests show up as status_LLM values other than 'ok' in step 5.
    print("\nWaiting for batch(es) to complete")

    pending = set(batch_ids)
    while pending:
        for bid in list(pending):
            batch = client.messages.batches.retrieve(bid)
            print(f"Batch {bid}: {batch.processing_status}   Counts: {batch.request_counts}")
            if batch.processing_status == "ended":
                pending.discard(bid)
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)

    # 5. Retrieve and parse results from every batch
    print("\nRetrieving results")

    # every result gets a status_LLM so problems stay visible in the data instead of being silently
    # dropped: 'ok' = tool call parsed, 'parse_error' = answer without a tool call, anything else =
    # the request itself failed (errored/expired/cancelled). The review notebook uses this column.
    raw_results = []
    for bid in batch_ids:
        for result in client.messages.batches.results(bid):
            # undo the custom_id conversion from the submission step
            pub_id = result.custom_id.replace("_", ".")
            if result.result.type == "succeeded":
                content = result.result.message.content
                stop_reason = result.result.message.stop_reason
                tool_block = next((b for b in content if b.type == "tool_use"), None)
                if tool_block:
                    record = dict(tool_block.input)
                    record["id"] = pub_id
                    record["status_LLM"] = "ok"
                    record["stop_reason_LLM"] = stop_reason
                else:
                    record = {"id": pub_id, "status_LLM": "parse_error", "stop_reason_LLM": stop_reason}
            else:
                record = {"id": pub_id, "status_LLM": result.result.type, "stop_reason_LLM": None}
            raw_results.append(record)

    results_df = pd.DataFrame(raw_results)
    # every field coming from the LLM gets a '_LLM' suffix, so it is always visible in the database
    # which columns are model output and which are curated or Dimensions data
    non_llm_cols = {"id", "status_LLM", "stop_reason_LLM"}
    results_df = results_df.rename(columns={c: f"{c}_LLM" for c in results_df.columns if c not in non_llm_cols})

    n_ok = (results_df["status_LLM"] == "ok").sum()
    n_missing_scope = results_df["scope_LLM"].isna().sum() if "scope_LLM" in results_df.columns else len(results_df)
    print(f"Results: {len(results_df)} total, {n_ok} succeeded, {n_missing_scope} missing a scope decision.")
    if "scope_LLM" in results_df.columns:
        print(f"LLM predicted in scope: {(results_df['scope_LLM'] == 'in').sum()}")

    # Derive pillar_LLM from the boolean flags (same logic as genai_scope_batchoutput.ipynb):
    # CC if multiple pillar flags are True, or if only cross_cutting_LLM is True
    # i.e. the four booleans are collapsed into ONE pillar per publication, because that is what the
    # dataset needs. Reading order: several pillars -> CC (cross-cutting), exactly one -> that one,
    # none -> 'NA'. Patents and Funding use exactly the same rule - keep them in sync.
    # Rows whose status_LLM is not 'ok' get None instead of a pillar, so a failed request is never
    # mistaken for "no pillar found".
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

    # 6. Write results back to RUN_TABLE
    # (and to CLASSIFICATION_TABLE - the loop further down runs over both tables)
    print(f"\nUpdating '{RUN_TABLE}' with LLM columns.")

    # the column types are stated explicitly because the columns are created via ALTER TABLE.
    # confidence_LLM is DOUBLE rather than INTEGER so a missing value can be stored as NULL.
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

    # the same update is applied to both tables: the run table documents this run, the classification
    # table is the one the review notebook (S5) and the labelling script (S6) read from
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


# only executed when the file is started directly (python S3_LLM_scope.py)
if __name__ == '__main__':
    main()
