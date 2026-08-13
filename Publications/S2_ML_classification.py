## STEP 2 of the publications pipeline
## Classification script: embed new input data and run through trained classifiers
##
## What it does: the fast, cheap pre-filter. Every publication from the run table is embedded with
## SPECTER2 and run through the two classifiers trained in S0_ML_training.py. The result is
## 'pred_combined' (1 = possibly in scope, 0 = out), which decides which publications are sent on to
## the much more expensive LLM step S3. The rule is deliberately generous: a publication counts as
## in scope if EITHER the scope classifier says in scope OR the pillar classifier assigns a pillar
## other than 'NA' - false positives are cheap (the LLM sorts them out), false negatives are not
## (they never appear in the dataset).
##
## Input:   RUN_TABLE (written by S1_query_dimensions.py) and the model files from S0
## Output:  RUN_TABLE gets the prediction columns; the rows are appended to CLASSIFICATION_TABLE;
##          the embeddings are stored in EMBEDDINGS_TABLE for reuse
##
## The script is written to be restartable: rows that already have embeddings, predictions, or are
## already in CLASSIFICATION_TABLE are recognised and skipped, so it can be re-run after a crash
## without doing (or paying for) anything twice.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import importlib.util, os

# torch and numpy/MKL each load their own OpenMP runtime on Windows, which can crash with
# "libomp.dll already initialized" when both end up in the same process. Must be set before
# torch/transformers are imported (they're imported lazily inside main(), but this still needs
# to run first since it's a process-wide env var, not something scoped to that import).
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# CONFIG
# edit parameters for this run here

# NOTE: started through pipeline_publications.py, all of these are overridden by the pipeline's own
# CONFIG - the defaults here only apply when the script is run on its own.

# Database
# path to DuckDB database
DB_PATH = 'publications.db'
# table containing the new input data to classify with run date as name
# the table S1_query_dimensions.py wrote (for a pipeline run: run_<date>_<time>, or its _reverse
# counterpart - the pipeline calls this script twice, once for each)
RUN_TABLE = 'data_run_test'
# table for embeddings
# permanent store of all computed embedding vectors, so no publication is ever embedded twice
EMBEDDINGS_TABLE = 'publications_embeddings'
# table for final classifications
# the accumulating main table; the classified rows of this run are appended to it
CLASSIFICATION_TABLE = 'publications_classified'

# Columns
# columns concatenated for embedding (title, abstract)
# must be exactly the same combination as in S0_ML_training.py, otherwise the models see a
# different kind of text than they were trained on
TEXT_COLUMNS = ('title', 'abstract')

# Embeddings
# save path for embeddings (checkpoint + final)
# .npy checkpoint file written after every batch so an interrupted run resumes instead of starting
# over. The pipeline gives every run its own file (embeddings/embeddings_<run table>.npy); when
# running standalone, use a fresh name per run - see the stale-checkpoint note in get_embeddings().
EMBEDDINGS_PATH = 'embeddings_run_test.npy'

# Model paths
# the files S0_ML_training.py produced. Both classifiers must come from the same training run as
# the threshold file, otherwise the threshold does not match the model's probabilities.
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'

# START OF SCRIPT

def main(
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    TEXT_COLUMNS=TEXT_COLUMNS,
    EMBEDDINGS_PATH=EMBEDDINGS_PATH,
    SCOPE_MODEL_PATH=SCOPE_MODEL_PATH,
    PILLAR_MODEL_PATH=PILLAR_MODEL_PATH,
    EMBEDDINGS_TABLE=EMBEDDINGS_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    THRESHOLD_PATH=THRESHOLD_PATH,
):
    # 1. Load ML_pipeline_functions from the same directory as this script
    # loaded by file path rather than a normal import so the script also works from another working
    # directory; mlf then holds the embedding/classification functions of that module
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        'ML_pipeline_publications_functions',
        os.path.join(_script_dir, 'ML_pipeline_publications_functions.py')
    )
    mlf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mlf)

    import duckdb
    import numpy as np
    import pandas as pd
    import json
    from datetime import datetime

    # 2. Load input data
    # everything from here to step 3 is only about not repeating work: the three filter steps below
    # (already classified / already predicted / already embedded) let this script be re-run safely
    # after a crash, and they are the reason a re-run often reports "no new rows to classify".
    db = duckdb.connect(database=DB_PATH)
    data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()

    # store original columns for saving the table later
    # noted before the working columns (text, embeddings, predictions) are added below
    original_columns = list(data.columns)
    print(f"{len(data)} rows loaded from '{RUN_TABLE}'")

    # skip rows already fully classified: handles resuming after a crash that happened after
    # RUN_TABLE was updated (step 7) but before the append to CLASSIFICATION_TABLE (step 8)
    # completed in a previous attempt
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        classified_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        n_before = len(data)
        data = data[~data['id'].isin(classified_ids)].reset_index(drop=True)
        print(f"{n_before - len(data)} rows already in {CLASSIFICATION_TABLE}, skipped.")

    if len(data) == 0:
        print("No new rows to classify.")
        db.close()
        return

    # rows that already have predictions (RUN_TABLE was updated in a previous attempt that
    # then crashed before appending to CLASSIFICATION_TABLE) can skip straight to the append
    # step below, rather than being re-embedded/re-classified or silently dropped
    if 'pred_combined' in data.columns:
        already_predicted = data[data['pred_combined'].notna()].reset_index(drop=True)
        data = data[data['pred_combined'].isna()].reset_index(drop=True)
    else:
        already_predicted = data.iloc[0:0].copy()
    if len(already_predicted) > 0:
        print(f"{len(already_predicted)} rows already classified in a previous attempt, appending directly.")

    if len(data) > 0:
        # remove entries with existing embedding
        existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
        if EMBEDDINGS_TABLE in existing_tables:
            existing_ids = db.sql(f"SELECT id FROM {EMBEDDINGS_TABLE}").df()['id']
            n_before = len(data)
            data = data[~data['id'].isin(existing_ids)].reset_index(drop=True)
            print(f"{n_before - len(data)} rows already in {EMBEDDINGS_TABLE}, skipped.")

        print(f"{len(data)} rows remaining for embedding.")

    # nothing left to embed/classify; also avoids get_embeddings()'s file checkpoint (which
    # tracks progress by row count, not id) silently returning stale cached rows for an empty input
    if len(data) == 0 and len(already_predicted) == 0:
        print("No new rows to classify.")
        db.close()
        return

    if len(data) > 0:
        # Build text column from title + abstract
        # same '[SEP]' construction as in S0_ML_training.py - it has to match the training input
        data['text'] = data[TEXT_COLUMNS[0]].fillna('') + ' [SEP] ' + data[TEXT_COLUMNS[1]].fillna('')

        # Load SPECTER2 model and check token sizes
        # the model is only loaded once there is actually something to embed (loading takes a while)
        print("\nLoading SPECTER2 model.")

        model, tokenizer = mlf.load_specter()
        # informational only: adds a 'truncated' column marking texts longer than 512 tokens.
        # Those are cut off by the model, which mainly affects very long abstracts.
        mlf.check_token_size(data, text_column='text', tokenizer=tokenizer, add_column=True)

        # 3. Compute embeddings
        # by far the slowest step of this script; uses the GPU if available. Progress is written to
        # EMBEDDINGS_PATH after every batch, so an interrupted run continues where it stopped.
        print("\nComputing embeddings.")

        embeddings = mlf.get_embeddings(
            data,
            text_column='text',
            file_path=EMBEDDINGS_PATH,
            model=model,
            tokenizer=tokenizer,
            checkpoint=True
        )
        print(f"Embeddings shape: {embeddings.shape}")

        data['embeddings'] = list(embeddings)

        # Save embeddings to database
        print(f"\nAppending to {EMBEDDINGS_TABLE} table.")

        data_embeddings = data[['id', 'embeddings']]
        db.register('data_embeddings', data_embeddings)

        existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
        if EMBEDDINGS_TABLE in existing_tables:
            db.sql(f"INSERT INTO {EMBEDDINGS_TABLE} SELECT * FROM data_embeddings")
        else:
            db.sql(f"CREATE TABLE {EMBEDDINGS_TABLE} AS SELECT * FROM data_embeddings")

        # 4. Run scope classifier
        print("\nRunning scope classifier.")

        # the threshold comes from training (S0) and is not stored inside the model file: a
        # publication counts as in scope when its probability is >= threshold. It is usually well
        # below 0.5, because training accepted at most MAX_FN false negatives.
        with open(THRESHOLD_PATH, 'r') as f:
            threshold = float(f.read().strip())
        print(f"Scope threshold: {threshold:.3f}")

        proba_scope, pred_scope = mlf.scope_classification(embeddings, SCOPE_MODEL_PATH, threshold=threshold)

        data['proba_scope']     = proba_scope
        data['pred_scope']      = pred_scope
        data['threshold_scope'] = threshold

        # 5. Run pillar classifier
        # predicts one of PB / F / CM / CC / NA; proba_pillar is the probability of that class only
        print("Running pillar classifier.")

        proba_pillar, pred_pillar = mlf.pillar_classification(embeddings, PILLAR_MODEL_PATH)

        data['proba_pillar'] = proba_pillar
        data['pred_pillar']  = pred_pillar

        # 6. Combine predictions
        # OR logic: in scope if the scope classifier says so, or if a pillar other than 'NA' was
        # assigned. Both prediction columns are kept in the database so the review notebook (S5) can
        # look at them separately - it uses proba_scope and pred_pillar for its decision rules.
        data['pred_combined'] = mlf.combine_classifications(pred_scope, pred_pillar)
        # date of the ML classification (yymmdd); useful to select rows of one particular run later
        data['date_ML'] = datetime.today().strftime('%y%m%d')

        in_scope_n = data['pred_combined'].sum()

        print(f"Predicted in scope: {in_scope_n} / {len(data)} ({in_scope_n / len(data):.1%})")

        # 7. Update RUN_TABLE in database with new prediction columns
        # written to the run table first, then appended to CLASSIFICATION_TABLE in step 8. If the
        # script dies in between, the resume logic at the top of the script recognises that.
        print(f"\nUpdating '{RUN_TABLE}' in database with prediction columns.")

        # ADD COLUMN IF NOT EXISTS + UPDATE instead of rewriting the table: the columns are created
        # once (with the types given here) and only the rows of this run are filled in
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

        db.register('data_predictions', data[['id'] + list(new_columns.keys())])
        db.sql(f"""
            UPDATE {RUN_TABLE}
            SET proba_scope     = data_predictions.proba_scope,
                pred_scope      = data_predictions.pred_scope,
                threshold_scope = data_predictions.threshold_scope,
                proba_pillar    = data_predictions.proba_pillar,
                pred_pillar     = data_predictions.pred_pillar,
                pred_combined   = data_predictions.pred_combined,
                embeddings      = data_predictions.embeddings,
                date_ML         = data_predictions.date_ML
            FROM data_predictions
            WHERE {RUN_TABLE}.id = data_predictions.id
        """)

    # 8. Append to publications_classified
    print(f"\nAppending to {CLASSIFICATION_TABLE} table.")

    # combine rows classified just now with any rows already classified in a previous attempt
    data_to_append = pd.concat([already_predicted, data], ignore_index=True) if len(already_predicted) > 0 else data

    # get output columns for CLASSIFICATION_TABLE
    # excludes columns computed later in the pipeline (S3_LLM_scope.py, S6_LLM_labelling.py)
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        output_columns = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                           if c not in ('scope_LLM', 'confidence_LLM', 'pillar_LLM', 'plant_based_LLM',
                                        'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
                                        'status_LLM', 'stop_reason_LLM',
                                        'date_LLM', 'date_labelling')]
    else:
        output_columns = original_columns + ['proba_scope', 'pred_combined', 'pred_pillar', 'date_ML']

    # convert prediction in / out and add to CLASSIFICATION_TABLE
    # note the difference in representation: pred_combined is 0/1 in RUN_TABLE but 'in'/'out' in
    # CLASSIFICATION_TABLE. Both spellings appear downstream - S3_LLM_scope.py filters the run table
    # on pred_combined == 1, the review notebook filters the classification table on 'in'/'out'.
    data_classified = data_to_append.reindex(columns=output_columns).copy()
    data_classified['pred_combined'] = data_classified['pred_combined'].map({1: 'in', 0: 'out'})

    # serialize nested Dimensions fields to JSON text: their native shape (STRUCT vs MAP) can
    # differ between queries depending on what data Dimensions returns, which crashes a plain
    # INSERT into the already-typed CLASSIFICATION_TABLE; JSON text is stable regardless of shape
    def _to_json(x):
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return json.dumps(x, default=str)

    for col in ('authors', 'funder_countries', 'research_org_cities', 'research_org_countries'):
        if col in data_classified.columns:
            data_classified[col] = data_classified[col].apply(_to_json)

    db.register('data_classified', data_classified)

    if CLASSIFICATION_TABLE in existing_tables:
        cols_sql = ", ".join(f'"{c}"' for c in output_columns)
        db.sql(f"INSERT INTO {CLASSIFICATION_TABLE} ({cols_sql}) SELECT * FROM data_classified")
    else:
        db.sql(f"CREATE TABLE {CLASSIFICATION_TABLE} AS SELECT * FROM data_classified")

    print(f"{len(data_classified)} rows appended to {CLASSIFICATION_TABLE}.")

    db.close()
    print("\nDone!")


# only executed when the file is started directly (python S2_ML_classification.py)
if __name__ == '__main__':
    main()
