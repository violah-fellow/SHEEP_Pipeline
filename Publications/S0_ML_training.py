## STEP 0 of the publications pipeline - the only step that is NOT part of a normal run
## Training script: embed new training data and train LR classifiers (scope + pillar)
##
## What it does: takes manually labelled publications (a human decided in/out of scope and which
## alternative-protein pillar they belong to), turns their title+abstract into SPECTER2 embeddings
## and trains the two classifiers that S2_ML_classification.py later applies to new data:
##   1. scope classifier  - binary, in scope vs out of scope
##   2. pillar classifier - multiclass, PB / F / CM / CC / NA
##
## Input:   publications_training.db, table TRAINING_TABLE with columns 'scope' and 'pillar'
## Output:  two joblib model files plus a threshold txt file in Models/ (see paths below)
##
## When to run it: only when the models should be (re)trained - e.g. after adding new labelled
## data or changing the scope definition. A normal pipeline run reuses the saved model files and
## never calls this script; pipeline_publications.py does not include it.
##
## Embeddings are cached in the training database, so re-running after adding new labelled rows only
## embeds the new rows but always retrains on everything labelled so far.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import importlib.util, sys, os

# CONFIG 
# edit parameters for this run here

# Database
# path to DuckDB database
# training data lives in its OWN database, separate from publications.db - the pipeline's working
# data and the labelled ground truth are deliberately kept apart
DB_PATH = 'publications_training.db'
# table containing new training data
# needs the columns used below: id, title, abstract, year, scope, pillar, research_category
TRAINING_TABLE = 'publications_new_training'
# table for embeddings
# cache of already computed embeddings inside the training database. Rows that are already in here
# are not embedded again (embedding is the slow part), but training always uses the full table.
EMBEDDINGS_TABLE = 'publications_embedding'

# Columns
# columns concatenated for embedding (title, abstract)
# joined as 'title [SEP] abstract' below - [SEP] is the separator token SPECTER2 was trained with,
# and the same combination has to be used in S2_ML_classification.py or predictions degrade
TEXT_COLUMNS = ('title', 'abstract')
# column with scope labels ('in' / 'out')
# converted to 1/0 further down; anything that is not exactly 'in' counts as out of scope
SCOPE_COLUMN = 'scope'
# column with pillar labels
# expected values: PB (plant-based), F (fermentation), CM (cultivated meat), CC (cross-cutting),
# NA (no pillar). 'NA' has to be present in the training data - S2 uses it as the "no pillar" class.
PILLAR_COLUMN = 'pillar'

# Embeddings
# checkpoint + final save path for embeddings
# a .npy file written after every batch, so an interrupted training run resumes instead of starting
# over. Delete this file when you start embedding a different set of rows (see get_embeddings()).
EMBEDDINGS_PATH = 'embeddings_new_training.npy'

# Model save paths
# these three files are what S2_ML_classification.py loads - overwriting them changes the
# classification of all future runs, so keep a copy of the old ones if you want to compare
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
# the threshold determined during training is stored as plain text and reused for classification
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'

# Classifier hyperparameters (passed as kwargs to train_scope / train_pillar)
# C = inverse regularisation strength (smaller = stronger regularisation),
# class_weight='balanced' compensates for the much larger number of out-of-scope examples
SCOPE_MODEL_KWARGS  = {'C': 0.1, 'class_weight': 'balanced'}
PILLAR_MODEL_KWARGS = {'C': 0.1, 'class_weight': 'balanced'}
# maximum share of false negatives accepted when the scope threshold is chosen: the threshold is
# lowered until fewer than 1% of the labelled in-scope publications would be missed. Deliberately
# strict, because the LLM step afterwards can still discard false positives - but anything dropped
# here is gone from the dataset for good.
MAX_FN              = 0.01

# START OF SCRIPT

def main(
    DB_PATH=DB_PATH,
    TRAINING_TABLE=TRAINING_TABLE,
    TEXT_COLUMNS=TEXT_COLUMNS,
    SCOPE_COLUMN=SCOPE_COLUMN,
    PILLAR_COLUMN=PILLAR_COLUMN,
    EMBEDDINGS_PATH=EMBEDDINGS_PATH,
    SCOPE_MODEL_PATH=SCOPE_MODEL_PATH,
    PILLAR_MODEL_PATH=PILLAR_MODEL_PATH,
    THRESHOLD_PATH=THRESHOLD_PATH,
    MAX_FN=MAX_FN,
    SCOPE_MODEL_KWARGS=SCOPE_MODEL_KWARGS,
    PILLAR_MODEL_KWARGS=PILLAR_MODEL_KWARGS,
):
    # Load ML_pipeline_functions from the same directory as this script
    # loaded by file path instead of a normal import so the script also works when it is started
    # from another working directory (e.g. from the repository root); mlf then holds all the
    # embedding/training functions documented in ML_pipeline_publications_functions.py
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        'ML_pipeline_publications_functions',
        os.path.join(_script_dir, 'ML_pipeline_publications_functions.py')
    )
    mlf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mlf)

    import duckdb
    import numpy as np

    # 1. Load training data & remove entries already in publications_embeddings
    print("Loading training data.")

    db = duckdb.connect(database=DB_PATH)
    data = db.sql(f"SELECT * FROM {TRAINING_TABLE}").df()
    print(f"{len(data)} rows loaded from '{TRAINING_TABLE}'")

    # rows whose embedding already exists are skipped - only genuinely new labelled data is embedded.
    # Adding new training data therefore means: add rows to TRAINING_TABLE and run this script again.
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if EMBEDDINGS_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {EMBEDDINGS_TABLE}").df()['id']
        n_before = len(data)
        data = data[~data['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(data)} rows already in {EMBEDDINGS_TABLE}, skipped.")
    print(f"{len(data)} rows remaining for embedding.")

    # Build text column from title + abstract
    # fillna('') so publications without an abstract still produce a usable text instead of NaN
    data['text'] = data[TEXT_COLUMNS[0]].fillna('') + ' [SEP] ' + data[TEXT_COLUMNS[1]].fillna('')

    # 2. Load SPECTER2 model
    # SPECTER2 is an embedding model trained on scientific literature; it is downloaded from
    # Hugging Face on first use and cached locally afterwards
    print("\nLoading SPECTER2 model.")

    model, tokenizer = mlf.load_specter()
    # only reports how many texts exceed the 512-token limit and marks them in a 'truncated' column.
    # Truncation itself is handled by the model - long abstracts are cut off, not rejected.
    mlf.check_token_size(data, text_column='text', tokenizer=tokenizer, add_column=True)

    # 3. Compute embeddings
    # the slow step: runs on the GPU if one is available, otherwise on the CPU. checkpoint=True
    # writes EMBEDDINGS_PATH after every batch so an interrupted run can be resumed.
    print("\nComputing embeddings.")

    embeddings = mlf.get_embeddings(
        data,
        text_column='text',
        file_path=EMBEDDINGS_PATH,
        model=model,
        tokenizer=tokenizer,
        checkpoint=True
    )

    # 4. Append new rows to {EMBEDDINGS_TABLE}
    print(f"\nAppending to {EMBEDDINGS_TABLE} table.")

    # the embeddings table keeps the labels next to the vectors, which is what makes step 5 possible:
    # training can be repeated on all data ever labelled without re-reading or re-embedding anything.
    # These column names must exist in TRAINING_TABLE - a missing one (e.g. research_category) fails here.
    EMBEDDINGS_COLUMNS = ['id', 'title', 'abstract', 'year', 'scope', 'pillar', 'research_category', 'truncated', 'embedding']

    data['embedding'] = list(embeddings)
    db.register('df_new', data[EMBEDDINGS_COLUMNS])

    if EMBEDDINGS_TABLE in existing_tables:
        db.sql(f"INSERT INTO {EMBEDDINGS_TABLE} SELECT * FROM df_new")
    else:
        db.sql(f"CREATE TABLE {EMBEDDINGS_TABLE} AS SELECT * FROM df_new")

    print(f"{len(data)} rows appended to {EMBEDDINGS_TABLE}.")

    # 5. Load all data from {EMBEDDINGS_TABLE} for training
    # deliberately the whole table, not just the newly added rows: the classifiers are always
    # retrained from scratch on every labelled publication collected so far
    print(f"\nLoading all training data from {EMBEDDINGS_TABLE}.")

    all_data = db.sql(f"SELECT * FROM {EMBEDDINGS_TABLE}").df()
    print(f"{len(all_data)} rows loaded for training.")

    all_embeddings = np.array(all_data['embedding'].tolist())
    # the scope classifier needs 0/1 instead of 'in'/'out'
    all_data['scope_binary'] = (all_data[SCOPE_COLUMN] == 'in').astype(int)

    # sanity check before training: with a strongly imbalanced or tiny class the resulting model is
    # unreliable, and a pillar barely represented in the labels will hardly ever be predicted

    print(f"Scope label distribution:\n{all_data['scope_binary'].value_counts().to_string()}")
    print(f"Pillar label distribution:\n{all_data[PILLAR_COLUMN].value_counts().to_string()}")

    # 6. Train scope classifier
    # train_scope() also determines the decision threshold: it uses 5-fold cross-validation to get
    # out-of-fold probabilities and picks the lowest threshold at which the false-negative rate stays
    # below MAX_FN. The printed threshold and FN rate are worth checking - a very low threshold means
    # a lot of records will be passed on to the LLM step.
    print("\nTraining scope classifier.")

    classifier_scope, threshold = mlf.train_scope(
        all_embeddings,
        all_data['scope_binary'].values,
        max_fn=MAX_FN,
        model_path=SCOPE_MODEL_PATH,
        **SCOPE_MODEL_KWARGS
    )

    # the threshold is stored separately because it is not part of the joblib model file;
    # S2_ML_classification.py reads it back from here
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(threshold))

    # 7. Train pillar classifier
    # no threshold here - the pillar is simply the class with the highest probability (argmax)
    print("\nTraining pillar classifier.")

    mlf.train_pillar(
        all_embeddings,
        all_data[PILLAR_COLUMN].values,
        model_path=PILLAR_MODEL_PATH,
        **PILLAR_MODEL_KWARGS
    )

    db.close()

    print("\nDone!")


# only executed when the file is started directly (python S0_ML_training.py)
if __name__ == '__main__':
    main()
