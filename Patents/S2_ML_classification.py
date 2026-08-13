## STEP 2 of the patents pipeline
## Classification script: embed new input data and run through trained classifiers
##
## What it does: the fast, cheap pre-filter. Patents are embedded with PatentSBERTa and run through
## the two classifiers from S0_ML_training.py; the result is 'pred_combined', which decides what goes
## on to the expensive LLM step (S3). In scope if EITHER the scope classifier says so OR the pillar
## classifier assigns a pillar other than 'NA'.
##
## The key difference from the publications version is that everything happens PER FAMILY - and there
## are two cases:
##   Known families   - a patent of that family has been classified before: its predictions are
##                      copied across, no embedding, no model, no cost.
##   New families     - one representative is chosen per family (preferably one with title and
##                      abstract, from WO/EP/US, newest), only that one is embedded and classified,
##                      and its result is propagated to every family member.
## This is what makes the step affordable: one classification per invention instead of one per
## document. The price is that all members of a family always share the same verdict.
##
## Input:   RUN_TABLE (written by S1_query_dimensions.py) and the model files from S0
## Output:  prediction columns in RUN_TABLE, the rows appended to CLASSIFICATION_TABLE, and the
##          representatives' embeddings in EMBEDDINGS_TABLE
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import importlib.util, os

# CONFIG
# edit parameters for this run here
# NOTE: started through pipeline_patents.py, these are overridden by the pipeline's own CONFIG.

# Database
# path to DuckDB database
DB_PATH = 'patents.db'
# table containing the new input data to classify with run date as name
RUN_TABLE = 'data_run_test'
# table for embeddings
# only ever holds the family representatives - the other family members inherit their predictions
# and are never embedded
EMBEDDINGS_TABLE = 'patents_embeddings'
# table for final classifications
# the accumulating main table; the classified rows of this run are appended to it
CLASSIFICATION_TABLE = 'patents_classified'

# Columns
# columns concatenated for embedding (title, abstract)
# must be exactly the same combination as in S0_ML_training.py
TEXT_COLUMNS = ('title', 'abstract')

# Embeddings
# save path for embeddings (checkpoint + final)
# .npy checkpoint written after every batch; the pipeline gives every run its own file. When running
# standalone, use a fresh name per run - the checkpoint counts rows, it does not know which patents
# they belong to (see get_embeddings()).
EMBEDDINGS_PATH = 'embeddings_run_test.npy'

# Model paths
# the files produced by S0_ML_training.py. Despite the 'LR_' prefix these are SVC models for patents
# (the name was kept from the publications pipeline) - the code does not care, it just loads them.
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'

# START OF SCRIPT

def main(
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    EMBEDDINGS_TABLE=EMBEDDINGS_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    TEXT_COLUMNS=TEXT_COLUMNS,
    EMBEDDINGS_PATH=EMBEDDINGS_PATH,
    SCOPE_MODEL_PATH=SCOPE_MODEL_PATH,
    PILLAR_MODEL_PATH=PILLAR_MODEL_PATH,
    THRESHOLD_PATH=THRESHOLD_PATH,
):
    # torch and numpy/MKL each bring their own OpenMP runtime on Windows, which can crash with
    # "libomp.dll already initialized" when both end up in the same process. Set before torch and
    # the transformer libraries are loaded below.
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

    # 1. Load ML_pipeline_functions from the same directory as this script
    # loaded by file path rather than a normal import so the script also works from another working
    # directory; mlf then holds the embedding/classification functions of that module
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        'ML_pipeline_patents_functions',
        os.path.join(_script_dir, 'ML_pipeline_patents_functions.py')
    )
    mlf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mlf)

    import duckdb
    import numpy as np
    import pandas as pd
    from datetime import datetime

    # 2. Load input data
    db = duckdb.connect(database=DB_PATH)
    data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()
    original_columns = list(data.columns)
    print(f"{len(data)} rows loaded from '{RUN_TABLE}'")

    # Prediction columns to be added to RUN_TABLE
    # created up front (empty) for both cases below, because known and new families are written back
    # separately. ADD COLUMN IF NOT EXISTS makes this safe to re-run.
    new_columns = {
        'embeddings':      'DOUBLE[]',
        'truncated':       'BOOLEAN',
        'proba_scope':     'DOUBLE',
        'pred_scope':      'INTEGER',
        'threshold_scope': 'DOUBLE',
        'proba_pillar':    'DOUBLE',
        'pred_pillar':     'VARCHAR',
        'pred_combined':   'INTEGER',
        'date_ML':         'VARCHAR',
    }
    for col, dtype in new_columns.items():
        db.sql(f"ALTER TABLE {RUN_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype}")

    # 3. Split into known families (already classified) and new families
    # the central decision of this script: known families cost nothing (step 4), new ones are
    # embedded and classified (step 5). The split is purely by family_id.
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()

    if CLASSIFICATION_TABLE in existing_tables:
        known_family_ids = db.sql(
            f"SELECT DISTINCT family_id FROM {CLASSIFICATION_TABLE}"
        ).df()['family_id']
        known_mask = data['family_id'].isin(known_family_ids)
        known_data = data[known_mask].copy()
        new_data = data[~known_mask].reset_index(drop=True)
        print(
            f"{len(known_data)} rows from {known_data['family_id'].nunique()} known families, "
            f"predictions will be added from {CLASSIFICATION_TABLE}.\n"
            f"{len(new_data)} rows from {new_data['family_id'].nunique()} new families, "
            f"will be embedded and classified."
        )
    else:
        known_data = pd.DataFrame()
        new_data = data.copy()
        print(f"No existing classification table. All {len(new_data)} rows treated as new.")

    # nested Dimensions fields (lists, lists of dicts) are stored as JSON text in
    # CLASSIFICATION_TABLE: their native shape can differ between queries, which makes a plain INSERT
    # into the already-typed table fail. JSON text is stable regardless of shape - but it means these
    # columns have to be parsed again for analysis (S5_Assignee_cleanup.py does exactly that).
    import json as _json
    def _to_json(x):
        if isinstance(x, str):
            return x
        if hasattr(x, '__len__'):
            return _json.dumps(x.tolist() if hasattr(x, 'tolist') else list(x), ensure_ascii=False)
        if x is None or pd.isna(x):
            return x
        return _json.dumps(x, ensure_ascii=False)

    # 4. Known families: propagate pred_combined and pred_pillar from classification table
    # no model runs here at all - the verdict of the family is simply adopted. Note the consequence:
    # if the models are retrained, these patents keep the OLD family verdict; only entirely new
    # families are scored with the new models.
    if not known_data.empty:
        print(f"\nPropagating predictions for known families.")

        # pred_combined is stored as 'in'/'out' in CLASSIFICATION_TABLE — convert back to int
        # DISTINCT ON (family_id) takes an arbitrary row per family - fine here, because all members
        # of a family carry the same predictions by construction
        family_preds = db.sql(f"""
            SELECT DISTINCT ON (family_id)
                family_id,
                proba_scope,
                pred_pillar,
                date_ML,
                CASE WHEN pred_combined = 'in' THEN 1 ELSE 0 END AS pred_combined
            FROM {CLASSIFICATION_TABLE}
        """).df()

        _drop = [c for c in ['proba_scope', 'pred_combined', 'pred_pillar', 'date_ML'] if c in known_data.columns]
        known_data = known_data.drop(columns=_drop).merge(family_preds, on='family_id', how='left')

        db.register('known_update', known_data[['id', 'proba_scope', 'pred_combined', 'pred_pillar']])
        db.sql(f"""
            UPDATE {RUN_TABLE}
            SET proba_scope   = known_update.proba_scope,
                pred_combined = known_update.pred_combined,
                pred_pillar   = known_update.pred_pillar
            FROM known_update
            WHERE {RUN_TABLE}.id = known_update.id
        """)
        print(f"{len(known_data)} rows updated in '{RUN_TABLE}'.")

        # Append new patents from known families to CLASSIFICATION_TABLE
        # (the family is known but these specific patents weren't seen before)
        _llm_cols = {'scope_LLM', 'confidence_LLM', 'pillar_LLM', 'plant_based_LLM',
                     'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
                     'status_LLM', 'stop_reason_LLM',
                     'date_LLM', 'date_labelling'}
        output_columns = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                          if c not in _llm_cols]
        known_classified = known_data.reindex(columns=output_columns).copy()
        known_classified['pred_combined'] = known_classified['pred_combined'].map({1: 'in', 0: 'out'})
        _nested = ['cpc', 'inventor_names', 'assignee_names', 'assignee_cities',
                   'assignee_countries', 'funder_countries']
        for _c in _nested:
            if _c in known_classified.columns:
                known_classified[_c] = known_classified[_c].apply(_to_json)
        _existing_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        known_classified = known_classified[~known_classified['id'].isin(_existing_ids)].reset_index(drop=True)
        db.register('known_classified', known_classified)
        _cols_str = ", ".join(f'"{c}"' for c in output_columns)
        db.sql(f"INSERT INTO {CLASSIFICATION_TABLE} ({_cols_str}) SELECT * FROM known_classified")
        print(f"{len(known_classified)} rows from known families appended to {CLASSIFICATION_TABLE}.")

    # 5. New families: select representative per family, embed, classify, propagate
    if not new_data.empty:

        # Select one representative per family for embedding
        # the priority order matters and is used identically in S3_LLM_scope.py and
        # S7_LLM_labelling.py - the same document should represent a family everywhere:
        #   1. must have a title AND an abstract (without text there is nothing to classify)
        #   2. prefer jurisdictions WO/EP/US - these usually carry the fullest English text
        #   3. of those, the newest publication_year
        #   4. if no member has title+abstract at all: simply the newest one (it will be classified
        #      on an almost empty text, which is why such families are usually filtered out)
        def select_patent(group):
            preferred_jurisdictions = ['WO', 'EP', 'US']
            has_content = group[
                group['title'].notna() & group['abstract'].notna() &
                (group['title'] != '') & (group['abstract'] != '')
            ].copy()

            # if no patent with title or abstract exists in the family: choose the newest
            if has_content.empty:
                return group.sort_values('publication_year', ascending=False).head(1)
            
            # in patents with title and abstract, choose the one from the preferred jurisdictions and newest publication date
            has_content['_preferred'] = has_content['jurisdiction'].isin(preferred_jurisdictions)
            has_content = has_content.sort_values(
                ['_preferred', 'publication_year'], ascending=[False, False]
            )
            return has_content.drop_duplicates(subset=['title', 'abstract']).head(1).drop(columns='_preferred')

        reps = new_data.groupby('family_id', group_keys=True).apply(select_patent).reset_index(level=0).reset_index(drop=True)
        print(f"\n{len(reps)} representative patents selected from {new_data['family_id'].nunique()} new families.")

        # same '[SEP]' construction as during training - the models must see the same kind of text
        reps['text'] = reps[TEXT_COLUMNS[0]].fillna('') + ' [SEP] ' + reps[TEXT_COLUMNS[1]].fillna('')

        # Load PatentSBERTa model and check token sizes
        # PatentSBERTa is trained on patent text (the publications pipeline uses SPECTER2 instead);
        # downloaded from Hugging Face on first use and cached locally
        print("\nLoading PatentSBERTa model.")
        model = mlf.load_patentsberta()
        mlf.check_token_size(reps, text_column='text', model=model, add_column=True)

        # Compute embeddings
        print("\nComputing embeddings.")
        embeddings = mlf.get_embeddings(
            reps,
            text_column='text',
            file_path=EMBEDDINGS_PATH,
            model=model,
            checkpoint=True
        )
        print(f"Embeddings shape: {embeddings.shape}")
        reps['embeddings'] = list(embeddings)

        # Save representative embeddings to database
        print(f"\nAppending to {EMBEDDINGS_TABLE} table.")
        data_embeddings = reps[['id', 'embeddings']]
        db.register('data_embeddings', data_embeddings)
        if EMBEDDINGS_TABLE in existing_tables:
            db.sql(f"INSERT INTO {EMBEDDINGS_TABLE} SELECT * FROM data_embeddings")
        else:
            db.sql(f"CREATE TABLE {EMBEDDINGS_TABLE} AS SELECT * FROM data_embeddings")

        # Run scope classifier
        # the threshold comes from training and is not part of the model file: in scope if the
        # probability is >= threshold. Usually well below 0.5, because training accepted at most
        # MAX_FN false negatives.
        print("\nRunning scope classifier.")
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Scope threshold: {threshold:.3f}")

        proba_scope, pred_scope = mlf.scope_classification(embeddings, SCOPE_MODEL_PATH, threshold=threshold)
        reps['proba_scope']     = proba_scope
        reps['pred_scope']      = pred_scope
        reps['threshold_scope'] = threshold

        # Run pillar classifier
        print("Running pillar classifier.")
        proba_pillar, pred_pillar = mlf.pillar_classification(embeddings, PILLAR_MODEL_PATH)
        reps['proba_pillar'] = proba_pillar
        reps['pred_pillar']  = pred_pillar

        # Combine predictions
        # OR logic: in scope if the scope classifier says so, or if a pillar other than 'NA' was
        # assigned. Deliberately generous - the LLM step sorts out the false positives.
        reps['pred_combined'] = mlf.combine_classifications(pred_scope, pred_pillar)
        # date of the ML classification (yymmdd); useful for selecting one run's records later
        reps['date_ML'] = datetime.today().strftime('%y%m%d')
        in_scope_n = reps['pred_combined'].sum()
        print(f"Predicted in scope: {in_scope_n} / {len(reps)} ({in_scope_n / len(reps):.1%})")

        # Propagate predictions from representative to all members of the same family
        # the merge on family_id gives every member the representative's values - this is the step
        # that turns "one classification" into "the whole family classified"
        pred_cols = [c for c in new_columns if c in reps.columns]
        _drop = [c for c in pred_cols if c in new_data.columns]
        new_data = new_data.drop(columns=_drop).merge(reps[['family_id'] + pred_cols], on='family_id', how='left')

        # Update RUN_TABLE for new families
        print(f"\nUpdating '{RUN_TABLE}' with predictions for new families.")
        available_cols = [c for c in new_columns if c in new_data.columns]
        db.register('new_update', new_data[['id'] + available_cols])
        set_clause = ", ".join(f"{c} = new_update.{c}" for c in available_cols)
        db.sql(f"""
            UPDATE {RUN_TABLE}
            SET {set_clause}
            FROM new_update
            WHERE {RUN_TABLE}.id = new_update.id
        """)

        # Append new families to CLASSIFICATION_TABLE
        print(f"\nAppending to {CLASSIFICATION_TABLE} table.")
        existing_tables_now = db.sql("SHOW TABLES").df()['name'].tolist()
        _llm_cols = {'scope_LLM', 'confidence_LLM', 'pillar_LLM', 'plant_based_LLM',
                     'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
                     'status_LLM', 'stop_reason_LLM',
                     'date_LLM', 'date_labelling'}
        if CLASSIFICATION_TABLE in existing_tables_now:
            output_columns = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                              if c not in _llm_cols]
        else:
            _extra = [c for c in ['proba_scope', 'pred_combined', 'pred_pillar', 'date_ML']
                      if c not in original_columns]
            output_columns = original_columns + _extra

        # deduplicate while preserving order (guards against partial prior runs)
        _seen = set()
        output_columns = [c for c in output_columns if c not in _seen and not _seen.add(c)]

        # note the difference in representation: pred_combined is 0/1 in RUN_TABLE but 'in'/'out' in
        # CLASSIFICATION_TABLE. Both spellings appear downstream - S3_LLM_scope.py filters the run
        # table on == 1, the review notebook filters the classification table on 'in'/'out'.
        data_classified = new_data.reindex(columns=output_columns).copy()
        data_classified['pred_combined'] = data_classified['pred_combined'].map({1: 'in', 0: 'out'})
        _nested = ['cpc', 'inventor_names', 'assignee_names', 'assignee_cities',
                   'assignee_countries', 'funder_countries']
        for _c in _nested:
            if _c in data_classified.columns:
                data_classified[_c] = data_classified[_c].apply(_to_json)
        db.register('data_classified', data_classified)

        if CLASSIFICATION_TABLE in existing_tables_now:
            _cols_str = ", ".join(f'"{c}"' for c in output_columns)
            db.sql(f"INSERT INTO {CLASSIFICATION_TABLE} ({_cols_str}) SELECT * FROM data_classified")
        else:
            db.sql(f"CREATE TABLE {CLASSIFICATION_TABLE} AS SELECT * FROM data_classified")

        print(f"{len(data_classified)} rows appended to {CLASSIFICATION_TABLE}.")

    db.close()
    print("\nDone!")


# only executed when the file is started directly (python S2_ML_classification.py)
if __name__ == '__main__':
    main()
