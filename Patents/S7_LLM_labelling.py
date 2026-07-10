## LLM labelling script: sends in-scope, pillar-curated patents to Claude via the Batch API and
## stores research category, end product, ingredient, and fermentation subpillar labels in
## patents_labelled. Each label type is an independent sub-run with its own batch/metadata file,
## so a crash or slow batch on one type doesn't force resubmitting the others on the next run.
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
# table with in-scope, pillar-classified patents to label 
CLASSIFICATION_TABLE = 'patents_classified'
# table for the final clean labelled output
LABELLED_TABLE = 'patents_labelled'

# Curated columns
# these are added to CLASSIFICATION_TABLE by manual review (see S6_Review_classifications.ipynb)
SCOPE_COL = 'scope_curated'
PILLAR_COL = 'pillar_curated'

LLM_MODEL_LABEL = 'claude-sonnet-4-6'  # or 'claude-haiku-4-5' for cheap test runs
MAX_TOKENS = 150
TEMPERATURE = 0.0

# Batch
# directory for batch submission metadata, one file per label type per run 
BATCH_DIR = 'batch_jobs'
# how often to check whether a batch has finished
POLL_INTERVAL_SECONDS = 600
# identifies this run's batch metadata files. Leave as None: for each label type, the script first
# looks in BATCH_DIR for a run of that type that was submitted but never finished and resumes that,
# if none exists it creates a timestamped label and submits a new batch. 
# Specific value: force resuming (or starting) a specific run across all three label types.
RUN_LABEL = None

# Research categories allowed per pillar, used to build the category tool call's enum.
PB_CATS = ["Crop development", "Strain development", "Ingredient optimisation", "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition", "Other"]
F_CATS  = ["Feedstocks", "Target molecule selection", "Strain development", "Bioprocess design", "Ingredient optimisation", "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition", "Other"]
CM_CATS = ["Cell line development", "Cell culture media", "Bioprocess design", "Scaffolding", "End product formulation", "Food safety & quality", "Health & nutrition", "Other"]
CC_CATS = ["Bioprocess design", "Scaffolding", "Ingredient optimisation", "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition", "Other"]
PILLAR_CATS = {'PB': PB_CATS, 'F': F_CATS, 'CM': CM_CATS, 'CC': CC_CATS}
# recognised pillar_curated values
RECOGNISED_PILLARS = set(PILLAR_CATS.keys())

# End product and ingredient use a single prompt/category list across all pillars
END_PRODUCT_CATS = ["Meat", "Fish and seafood", "Milk and milk proteins", "Yoghurt and fermented dairy", "Cheese", "Cream and ice cream", "Agnostic", "Chocolate, desserts, and confectionery", "Eggs and egg proteins", "Cross-cutting", "Spreads, sauces, and condiments", "Dairy"]
INGREDIENT_CATS = ["Isolates, concentrates, and flours", "Emulsions, gels, and binders", "N/A", "Flavours and aromas", "Colours", "Fats and oils"]

# Subpillar (biomass vs precision fermentation) only applies to Fermentation-pillar patents;
# a single label, not a primary/secondary pair (see LABEL_TYPES below)
SUBPILLAR_CATS = ["BF", "PF", "NA"]

# sentinel group key for label types that use one prompt/category list for every pillar
_UNGROUPED = '_all_'

# One entry per label type. 'grouped_by_pillar' selects whether prompt_paths/categories are keyed
# by pillar_curated (PB/F/CM/CC) or by the single _UNGROUPED key applied to every row.
LABEL_TYPES = {
    'category': {
        'grouped_by_pillar': True,
        'prompt_paths': {
            'PB': 'llm_prompts/category_prompt_patents_PB.md',
            'F':  'llm_prompts/category_prompt_patents_F.md',
            'CM': 'llm_prompts/category_prompt_patents_CM.md',
            'CC': 'llm_prompts/category_prompt_patents_CC.md',
        },
        'categories': PILLAR_CATS,
        'output_column': 'research_category',
        'diagnostic_prefix': 'category',
        'tool_name': 'label_research_category',
        'tool_description': "Record the primary and secondary research category for a patent.",
    },
    'endproduct': {
        'grouped_by_pillar': False,
        'prompt_paths': {_UNGROUPED: 'llm_prompts/endproduct_prompt_patents.md'},
        'categories': {_UNGROUPED: END_PRODUCT_CATS},
        'output_column': 'end_product',
        'diagnostic_prefix': 'endproduct',
        'tool_name': 'label_end_product',
        'tool_description': "Record the primary and secondary end product category for a patent.",
    },
    'ingredient': {
        'grouped_by_pillar': False,
        'prompt_paths': {_UNGROUPED: 'llm_prompts/ingredient_prompt_patents.md'},
        'categories': {_UNGROUPED: INGREDIENT_CATS},
        'output_column': 'ingredient',
        'diagnostic_prefix': 'ingredient',
        'tool_name': 'label_ingredient_type',
        'tool_description': "Record the primary and secondary ingredient type for a patent.",
    },
    'subpillar': {
        'grouped_by_pillar': False,
        # single BF/PF/NA label, not a primary/secondary pair
        'has_secondary': False,
        # only Fermentation-pillar patents go through the LLM for this label; every other
        # pillar gets 'NA' written directly, with no API call
        'restrict_to_pillars': {'F'},
        'auto_value_for_other_pillars': 'NA',
        'prompt_paths': {_UNGROUPED: 'llm_prompts/subpillar_prompt_patents.md'},
        'categories': {_UNGROUPED: SUBPILLAR_CATS},
        'output_column': 'subpillar',
        'diagnostic_prefix': 'subpillar',
        'tool_name': 'label_subpillar',
        'tool_description': "Record the fermentation subpillar classification (BF, PF, or NA) for a patent.",
    },
}

# columns on CLASSIFICATION_TABLE that are ML/LLM working columns, not patent metadata,
# excluded when building LABELLED_TABLE rows. 
NON_METADATA_COLS = {
    'proba_scope', 'pred_combined', 'pred_pillar',
    'scope_LLM', 'confidence_LLM', 'pillar_LLM',
    'plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
    'status_LLM', 'stop_reason_LLM',
}
for _cfg in LABEL_TYPES.values():
    _prefix = _cfg['diagnostic_prefix']
    NON_METADATA_COLS |= {
        f'primary_{_prefix}_LLM', f'secondary_{_prefix}_LLM',
        f'{_prefix}_status_LLM', f'{_prefix}_stop_reason_LLM',
    }
del _cfg, _prefix

# nested Dimensions fields that need JSON serialization before being written to a new table
NESTED_JSON_COLS = ('assignee_cities', 'assignee_countries', 'publications')

# START OF SCRIPT

def _build_category_tool(categories, tool_name, tool_description, has_secondary=True):
    properties = {
        "primary": {
            "type": "string",
            "enum": categories,
            "description": (
                "The category that best captures the patent's primary focus."
                if has_secondary else
                "The classification that applies to this patent."
            ),
        },
    }
    required = ["primary"]
    if has_secondary:
        properties["secondary"] = {
            "type": "string",
            "enum": categories,
            "description": "The second most relevant category."
        }
        required.append("secondary")
    return {
        "name": tool_name,
        "description": tool_description,
        "input_schema": {
            "type": "object",
            "properties": properties,
            "required": required,
        }
    }


def _normalise_category(value, categories):
    # the LLM sometimes returns different casing (e.g. "Health & Nutrition") despite the enum
    # constraint; look up case-insensitively and treat anything else as unparseable
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
    LABEL_TYPES=LABEL_TYPES,
    RECOGNISED_PILLARS=RECOGNISED_PILLARS,
    LLM_MODEL_LABEL=LLM_MODEL_LABEL,
    MAX_TOKENS=MAX_TOKENS,
    TEMPERATURE=TEMPERATURE,
    BATCH_DIR=BATCH_DIR,
    POLL_INTERVAL_SECONDS=POLL_INTERVAL_SECONDS,
    RUN_LABEL=RUN_LABEL,
    NON_METADATA_COLS=NON_METADATA_COLS,
    NESTED_JSON_COLS=NESTED_JSON_COLS,
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

    def find_incomplete_batch(label_type):
        """Return the run_label of the most recent batch of this label type that was submitted
        but never finished writing results (no 'completed_at' in its metadata), or None."""
        files = sorted(
            (f for f in os.listdir(batch_dir) if f.endswith(f'_{label_type}.json')),
            reverse=True,
        )
        for fname in files:
            meta = json.loads((batch_dir / fname).read_text(encoding="utf-8"))
            if 'completed_at' not in meta:
                return meta['run_label']
        return None

    def write_label_diagnostics(db, results_df, primary_col, secondary_col, status_col,
                                 stopreason_col, has_secondary):
        """ALTER + UPDATE CLASSIFICATION_TABLE with this label type's diagnostic columns.
        Shared by both the LLM-results path and the auto-value path (e.g. subpillar's 'NA'
        for non-Fermentation pillars), so the two stay in sync. date_labelled is merged as
        the max of the old and new value, since label types can complete on different days."""
        llm_columns = {
            primary_col:    'VARCHAR',
            status_col:     'VARCHAR',
            stopreason_col: 'VARCHAR',
        }
        if has_secondary:
            llm_columns[secondary_col] = 'VARCHAR'
        results_df = results_df.reindex(columns=['id'] + list(llm_columns.keys()) + ['date_labelled'])

        for col, dtype in llm_columns.items():
            db.sql(f"ALTER TABLE {CLASSIFICATION_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype}")
        db.sql(f"ALTER TABLE {CLASSIFICATION_TABLE} ADD COLUMN IF NOT EXISTS date_labelled VARCHAR")

        db.register('llm_results', results_df)
        set_clause = ", ".join(f"{col} = llm_results.{col}" for col in llm_columns)
        set_clause += (
            f", date_labelled = GREATEST("
            f"COALESCE({CLASSIFICATION_TABLE}.date_labelled, llm_results.date_labelled), "
            f"llm_results.date_labelled)"
        )
        db.sql(f"""
            UPDATE {CLASSIFICATION_TABLE}
            SET {set_clause}
            FROM llm_results
            WHERE {CLASSIFICATION_TABLE}.id = llm_results.id
        """)
        return len(results_df)

    # 2. Run each label type's batch independently: submit-or-resume, poll, parse, write
    # diagnostic columns back to CLASSIFICATION_TABLE. A crash or slow batch on one label type
    # doesn't block or resubmit the others on the next call.
    for label_type, cfg in LABEL_TYPES.items():
        print(f"\n{'=' * 60}\nLabel type: {label_type}\n{'=' * 60}")

        prefix = cfg['diagnostic_prefix']
        primary_col    = f"primary_{prefix}_LLM"
        secondary_col  = f"secondary_{prefix}_LLM"
        status_col     = f"{prefix}_status_LLM"
        stopreason_col = f"{prefix}_stop_reason_LLM"

        if RUN_LABEL is not None:
            run_label = RUN_LABEL
        else:
            run_label = find_incomplete_batch(label_type)
        if run_label is None:
            run_label = f"labelling_{datetime.today().strftime('%y%m%d_%H%M')}"
            print(f"Starting new run '{run_label}' for '{label_type}'")
        else:
            print(f"Resuming run '{run_label}' for '{label_type}'")

        metadata_path = batch_dir / f"{run_label}_{label_type}.json"

        if metadata_path.exists():
            # A batch for this run/label type was already submitted; resume instead of resubmitting.
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            batch_id = metadata["batch_id"]
            print(f"Found existing batch for '{run_label}' ({label_type}): {batch_id}. Resuming without resubmitting.")
        else:
            with duckdb.connect(database=DB_PATH) as db:
                data = db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE}").df()
                print(f"{len(data)} rows loaded from '{CLASSIFICATION_TABLE}'")

                data = data[data[SCOPE_COL] == 'in'].reset_index(drop=True)
                print(f"{len(data)} rows in scope (curated).")

                n_before = len(data)
                data = data[data[PILLAR_COL].isin(RECOGNISED_PILLARS)].reset_index(drop=True)
                if n_before - len(data) > 0:
                    print(f"{n_before - len(data)} rows dropped for missing/unrecognised '{PILLAR_COL}'.")

                if status_col in data.columns:
                    n_before = len(data)
                    data = data[data[status_col] != 'ok'].reset_index(drop=True)
                    print(f"{n_before - len(data)} rows already labelled for '{label_type}', skipped.")

                # Some label types only apply to a subset of pillars (e.g. subpillar only makes
                # sense for Fermentation patents). Rows outside that subset get an automatic
                # value written directly, with no LLM call and no batch involvement at all.
                restrict_to = cfg.get('restrict_to_pillars')
                if restrict_to is not None:
                    auto_mask = ~data[PILLAR_COL].isin(restrict_to)
                    if auto_mask.any():
                        auto_value = cfg['auto_value_for_other_pillars']
                        has_secondary = cfg.get('has_secondary', True)
                        auto_results = pd.DataFrame({'id': data.loc[auto_mask, 'id']})
                        auto_results[primary_col] = auto_value
                        if has_secondary:
                            auto_results[secondary_col] = None
                        auto_results[status_col] = 'ok'
                        auto_results[stopreason_col] = None
                        auto_results['date_labelled'] = datetime.today().strftime('%y%m%d')
                        n_auto = write_label_diagnostics(
                            db, auto_results, primary_col, secondary_col, status_col,
                            stopreason_col, has_secondary,
                        )
                        print(f"{n_auto} rows outside {sorted(restrict_to)} auto-set to "
                              f"'{auto_value}' for '{label_type}' (no LLM call).")
                    data = data[~auto_mask].reset_index(drop=True)

            if len(data) == 0:
                print(f"No new rows to label for '{label_type}'.")
                continue

            # 3. Build and submit batch requests. Grouped label types (category) pick a
            # pillar-specific prompt/tool per row; ungrouped types (endproduct, ingredient) use
            # the single prompt/tool for every row.
            data['_group'] = data[PILLAR_COL] if cfg['grouped_by_pillar'] else _UNGROUPED

            system_prompts = {}
            tools = {}
            for group, prompt_path in cfg['prompt_paths'].items():
                with open(prompt_path, "r", encoding="utf-8") as f:
                    system_prompts[group] = f.read().strip()
                tools[group] = _build_category_tool(
                    cfg['categories'][group], cfg['tool_name'], cfg['tool_description'],
                    cfg.get('has_secondary', True),
                )

            def build_batch_request(row):
                group = row['_group']
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
                                "text": system_prompts[group],
                                "cache_control": {"type": "ephemeral"}
                            }
                        ],
                        "messages": [{"role": "user", "content": user_message}],
                        "tools": [tools[group]],
                        "tool_choice": {"type": "tool", "name": cfg['tool_name']},
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
                "run_label": run_label,
                "label_type": label_type,
                "model": LLM_MODEL_LABEL,
                "n_records": len(data),
                "dataset_ids": data["id"].tolist(),
                # persisted so the resume path can still normalise categories per group
                "group_by_id": dict(zip(data["id"], data['_group'])),
                "created_at": datetime.now().isoformat(),
            }
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"Batch metadata saved to {metadata_path}")

        group_by_id = metadata["group_by_id"]

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

        has_secondary = cfg.get('has_secondary', True)

        raw_results = []
        for result in client.messages.batches.results(batch_id):
            row_id = result.custom_id.replace("_", ".")
            categories = cfg['categories'].get(group_by_id.get(row_id), [])
            if result.result.type == "succeeded":
                content = result.result.message.content
                stop_reason = result.result.message.stop_reason
                tool_block = next((b for b in content if b.type == "tool_use"), None)
                primary = _normalise_category(tool_block.input.get("primary"), categories) if tool_block else None
                if primary is None:
                    record = {"id": row_id, "status": "parse_error", "stop_reason": stop_reason}
                else:
                    record = {"id": row_id, "primary": primary, "status": "ok", "stop_reason": stop_reason}
                    if has_secondary:
                        record["secondary"] = _normalise_category(tool_block.input.get("secondary"), categories)
            else:
                record = {"id": row_id, "status": result.result.type, "stop_reason": None}
            raw_results.append(record)

        results_df = pd.DataFrame(raw_results)
        results_df["date_labelled"] = datetime.today().strftime('%y%m%d')
        rename_map = {"primary": primary_col, "status": status_col, "stop_reason": stopreason_col}
        if has_secondary:
            rename_map["secondary"] = secondary_col
        results_df = results_df.rename(columns=rename_map)

        n_ok = (results_df[status_col] == "ok").sum()
        print(f"Results: {len(results_df)} total, {n_ok} succeeded.")

        # 6. Write diagnostic columns for this label type back to CLASSIFICATION_TABLE. 
        print(f"\nUpdating '{CLASSIFICATION_TABLE}' with {label_type} columns.")

        with duckdb.connect(database=DB_PATH) as db:
            n_written = write_label_diagnostics(
                db, results_df, primary_col, secondary_col, status_col, stopreason_col, has_secondary,
            )
            print(f"'{CLASSIFICATION_TABLE}' updated with {label_type} columns for {n_written} rows.")

        metadata['completed_at'] = datetime.now().isoformat()
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # 7. Assemble LABELLED_TABLE: only rows successfully labelled by every label type get added.
    print(f"\n{'=' * 60}\nAssembling '{LABELLED_TABLE}'\n{'=' * 60}")

    status_cols = {lt: f"{cfg['diagnostic_prefix']}_status_LLM" for lt, cfg in LABEL_TYPES.items()}
    primary_cols = {lt: f"primary_{cfg['diagnostic_prefix']}_LLM" for lt, cfg in LABEL_TYPES.items()}

    with duckdb.connect(database=DB_PATH) as db:
        all_ok = " AND ".join(f'"{col}" = \'ok\'' for col in status_cols.values())
        candidates = db.sql(f"""
            SELECT * FROM {CLASSIFICATION_TABLE}
            WHERE {SCOPE_COL} = 'in' AND {all_ok}
        """).df()
        print(f"{len(candidates)} rows fully labelled across all {len(LABEL_TYPES)} label types.")

        existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
        if LABELLED_TABLE in existing_tables:
            labelled_ids = set(db.sql(f"SELECT id FROM {LABELLED_TABLE}").df()['id'])
            n_before = len(candidates)
            candidates = candidates[~candidates['id'].isin(labelled_ids)].reset_index(drop=True)
            print(f"{n_before - len(candidates)} rows already in '{LABELLED_TABLE}', skipped.")
        else:
            print(f"'{LABELLED_TABLE}' does not exist yet, will be created.")

        if len(candidates) == 0:
            print(f"\nNo new fully-labelled rows to add to '{LABELLED_TABLE}'.")
            print("\nDone!")
            return

        dimensions_columns = [c for c in candidates.columns if c not in NON_METADATA_COLS]
        labelled_data = candidates[dimensions_columns].copy()
        for label_type, cfg in LABEL_TYPES.items():
            labelled_data[cfg['output_column']] = candidates[primary_cols[label_type]]

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
        if LABELLED_TABLE in existing_tables:
            cols_sql = ", ".join(f'"{c}"' for c in labelled_data.columns)
            db.sql(f"INSERT INTO {LABELLED_TABLE} ({cols_sql}) SELECT * FROM labelled_data")
        else:
            db.sql(f"CREATE TABLE {LABELLED_TABLE} AS SELECT * FROM labelled_data")
        print(f"'{LABELLED_TABLE}' now contains the newly labelled rows.")

    print("\nDone!")


if __name__ == '__main__':
    main()
