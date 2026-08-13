## STEP 0 of the patents pipeline - the only step that is NOT part of a normal run
## Training script: embed new training data and train LR classifiers (scope + pillar)
## (despite the name these are SVC models here, not logistic regressions - see the kwargs below)
##
## What it does: takes manually labelled patents (a human decided in/out of scope and the pillar),
## embeds title+abstract with PatentSBERTa and trains the two classifiers that
## S2_ML_classification.py applies to new data:
##   1. scope classifier  - binary, in scope vs out of scope
##   2. pillar classifier - multiclass, PB / F / CM / CC / NA
##
## Input:   patents_training.db, table TRAINING_TABLE with columns 'scope' and 'pillar'
## Output:  two joblib model files plus a threshold txt file in Models/
##
## When to run it: only when the models should be (re)trained. A normal pipeline run reuses the saved
## files; pipeline_patents.py does not call this script.
##
## Embeddings are cached in the training database, so re-running after adding new labelled rows only
## embeds the new rows but always retrains on everything labelled so far.
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

import importlib.util, sys, os

# CONFIG 
# edit parameters for this run here

# Database
# path to DuckDB database
# training data lives in its own database, separate from patents.db
DB_PATH = 'patents_training.db'
# table containing new training data
# needs the columns used further down, including 'scope' and 'pillar'
TRAINING_TABLE = 'patents_raw'
# table for embeddings
# cache of computed embeddings inside the TRAINING database. Careful: the same table name is used in
# patents.db by S2_ML_classification.py - different file, so they do not interfere, but do not mix
# the two up when inspecting the databases.
EMBEDDINGS_TABLE = 'patents_embeddings'

# Columns
# columns concatenated for embedding (title, abstract)
# joined as 'title [SEP] abstract'; S2_ML_classification.py must use the same combination
TEXT_COLUMNS = ('title', 'abstract')
# column with scope labels ('in' / 'out')
# converted to 1/0 below; anything that is not exactly 'in' counts as out of scope
SCOPE_COLUMN = 'scope'
# column with pillar labels
# expected: PB (plant-based), F (fermentation), CM (cultivated meat), CC (cross-cutting), NA
PILLAR_COLUMN = 'pillar'

# Embeddings
# checkpoint + final save path for embeddings
# written after every batch so an interrupted training run resumes; delete it before embedding a
# different set of rows
EMBEDDINGS_PATH = 'embeddings_training.npy'

# Model save paths
# what S2_ML_classification.py loads. Overwriting these changes the classification of all future
# runs - keep a copy of the old files if you want to compare.
SCOPE_MODEL_PATH  = 'models/LR_scope.joblib'
PILLAR_MODEL_PATH = 'models/LR_pillar.joblib'
THRESHOLD_PATH    = 'models/LR_scope_threshold.txt'

# Classifier hyperparameters (passed as kwargs to train_scope / train_pillar)
# these go to an SVC with an RBF kernel (not logistic regression as in the publications pipeline):
#   C            - how hard the model tries to fit the training data (higher = more complex boundary)
#   gamma        - reach of a single training example in the RBF kernel
#   class_weight - 'balanced' compensates for the much larger number of out-of-scope examples
#   max_iter     - accepted by the wrapper but not an SVC parameter in this configuration
SCOPE_MODEL_KWARGS  = {'C': 100, 'class_weight': 'balanced', 'max_iter': 1000, 'kernel': 'rbf', 'gamma': 0.01}
PILLAR_MODEL_KWARGS = {'C': 1000, 'class_weight': 'balanced', 'max_iter': 1000, 'kernel': 'rbf', 'gamma': 0.01}
# maximum share of false negatives accepted when choosing the scope threshold. Stricter than for
# publications (0.01) because a discarded patent family takes all its members out of the dataset
# at once - so the threshold is pushed lower and more records are passed on to the LLM.
MAX_FN              = 0.004

# START OF SCRIPT

def main(
    DB_PATH=DB_PATH,
    TRAINING_TABLE=TRAINING_TABLE,
    EMBEDDINGS_TABLE=EMBEDDINGS_TABLE,
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
    # loaded by file path rather than a normal import so the script also works from another working
    # directory; mlf then holds all embedding/training functions of that module
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _spec = importlib.util.spec_from_file_location(
        'ML_pipeline_patents_functions',
        os.path.join(_script_dir, 'ML_pipeline_patents_functions.py')
    )
    mlf = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(mlf)

    import duckdb
    import numpy as np

    # 1. Load training data & remove entries already in publications_embeddings
    # (the comment says publications - it means EMBEDDINGS_TABLE, i.e. patents_embeddings here)
    # rows already embedded are skipped; adding new training data therefore means: add rows to
    # TRAINING_TABLE and run this script again.
    print("Loading training data.")

    db = duckdb.connect(database=DB_PATH)
    data = db.sql(f"SELECT * FROM {TRAINING_TABLE}").df()
    print(f"{len(data)} rows loaded from '{TRAINING_TABLE}'")

    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if EMBEDDINGS_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {EMBEDDINGS_TABLE}").df()['id']
        n_before = len(data)
        data = data[~data['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(data)} rows already in {EMBEDDINGS_TABLE}, skipped.")
    print(f"{len(data)} rows remaining for embedding.")

    # Build text column from title + abstract
    # fillna('') so patents without an abstract still produce usable text instead of NaN
    data['text'] = data[TEXT_COLUMNS[0]].fillna('') + ' [SEP] ' + data[TEXT_COLUMNS[1]].fillna('')

    # 2. Load PatentSBERTa model
    # embedding model trained on patent text; downloaded from Hugging Face on first use
    print("\nLoading PatentSBERTa model.")

    model = mlf.load_patentsberta()
    # informational only: marks texts above the token limit in a 'truncated' column. Long patent
    # abstracts are cut off by the model, not rejected.
    mlf.check_token_size(data, text_column='text', model=model, add_column=True)

    # 3. Compute embeddings
    # the slow step; uses the GPU if available and writes a checkpoint after every batch
    print("\nComputing embeddings.")

    embeddings = mlf.get_embeddings(
        data,
        text_column='text',
        file_path=EMBEDDINGS_PATH,
        model=model,
        checkpoint=True
    )

    # 4. Append new rows to {EMBEDDINGS_TABLE}
    print(f"\nAppending to {EMBEDDINGS_TABLE} table.")

    # the labels are stored alongside the vectors - that is what makes step 5 possible: retraining on
    # everything ever labelled without re-reading or re-embedding anything
    EMBEDDINGS_COLUMNS = ['id', 'embeddings', 'scope', 'pillar']

    data['embeddings'] = list(embeddings)
    db.register('df_new', data[EMBEDDINGS_COLUMNS])

    if EMBEDDINGS_TABLE in existing_tables:
        db.sql(f"INSERT INTO {EMBEDDINGS_TABLE} SELECT * FROM df_new")
    else:
        db.sql(f"CREATE TABLE {EMBEDDINGS_TABLE} AS SELECT * FROM df_new")

    print(f"{len(data)} rows appended to {EMBEDDINGS_TABLE}.")

    # 5. Load all data from {EMBEDDINGS_TABLE} for training
    # deliberately the whole table: the classifiers are always retrained from scratch on every
    # labelled patent collected so far
    print(f"\nLoading all training data from {EMBEDDINGS_TABLE}.")

    all_data = db.sql(f"SELECT * FROM {EMBEDDINGS_TABLE}").df()
    print(f"{len(all_data)} rows loaded for training.")

    all_embeddings = np.array(all_data['embeddings'].tolist())
    # the scope classifier needs 0/1 instead of 'in'/'out'
    all_data['scope_binary'] = (all_data[SCOPE_COLUMN] == 'in').astype(int)

    # sanity check before training: a very imbalanced or tiny class makes the model unreliable, and a
    # pillar barely present in the labels will hardly ever be predicted

    print(f"Scope label distribution:\n{all_data['scope_binary'].value_counts().to_string()}")
    print(f"Pillar label distribution:\n{all_data[PILLAR_COLUMN].value_counts().to_string()}")

    # 6. Train scope classifier
    # train_scope() also determines the decision threshold: 5-fold cross-validation for out-of-fold
    # probabilities, then the lowest threshold at which the false-negative rate stays below MAX_FN.
    # The printed threshold and FN rate are worth checking - a very low threshold means a lot of
    # records will be handed to the LLM step.
    print("\nTraining scope classifier.")

    classifier_scope, threshold = mlf.train_scope(
        all_embeddings,
        all_data['scope_binary'].values,
        max_fn=MAX_FN,
        model_path=SCOPE_MODEL_PATH,
        **SCOPE_MODEL_KWARGS
    )

    # stored separately because it is not part of the joblib model file; S2_ML_classification.py
    # reads it back from here
    with open(THRESHOLD_PATH, 'w') as f:
        f.write(str(threshold))

    # 7. Train pillar classifier
    # no threshold here - the predicted pillar is simply the class with the highest probability.
    # fillna('NA'): patents without a pillar label are trained as the explicit "no pillar" class, so
    # the model can actively predict "not an alternative-protein patent" rather than being forced
    # into one of the four pillars.
    print("\nTraining pillar classifier.")

    mlf.train_pillar(
        all_embeddings,
        all_data[PILLAR_COLUMN].fillna('NA').values,
        model_path=PILLAR_MODEL_PATH,
        **PILLAR_MODEL_KWARGS
    )

    db.close()

    print("\nDone!")


# only executed when the file is started directly (python S0_ML_training.py)
if __name__ == '__main__':
    main()
