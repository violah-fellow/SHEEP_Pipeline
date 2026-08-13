## STEP 1 of the publications pipeline
## Query script: query dimensions and store queried data in database
##
## What it does: sends every search string in STRINGS_FILE to the Dimensions API, collects all
## matching publications of the configured years, throws away duplicates and everything that is
## not a journal article, drops publications that were already processed in an earlier run, and
## stores what is left as a new "run table" in the DuckDB database.
##
## Input:   dimensions_search_publications.txt (one DSL search string per line)
##          DIMENSIONS_API_KEY from the .env file
## Output:  RUN_TABLE in publications.db - the raw input for S2_ML_classification.py
##
## Run it either standalone (`python S1_query_dimensions.py`, uses the CONFIG values below) or
## through pipeline_publications.py, which imports main() and passes its own parameters instead.
## All parameters that need to be changed are defined in the CONFIG section below.

import os

# CONFIG 
# edit parameters for this run here
# NOTE: these values are only used when the script is run on its own. Started through
# pipeline_publications.py, every parameter below is overridden by the pipeline's own CONFIG -
# so for a pipeline run, edit it there and not here.

# Dimensions API
# path to API key
# the .env file holding DIMENSIONS_API_KEY; '../.env' is the one in the repository root
KEY_PATH = '../.env'

# Database
# path to DuckDB database
# a single file that holds every table of this pipeline (run tables, embeddings, classifications)
DB_PATH = 'publications.db'
# table to store the new run data
# the pipeline names this run_<date>_<time>, so every run keeps its own raw query results and stays
# traceable afterwards. A table of the same name is overwritten (see CREATE OR REPLACE below).
RUN_TABLE = 'data_run_test'
# table for final classifications
# the long-lived table the whole pipeline accumulates into, across all runs. Used in this script
# only to skip publications that an earlier run already retrieved.
CLASSIFICATION_TABLE = 'publications_classified'

# Queries
# path to txt file with search strings
# one Dimensions DSL search string per line; each line is queried separately and the results merged
STRINGS_FILE = 'dimensions_search_publications.txt'
# Other parameters for search
# publication years to search, inclusive on both ends - set both to the same value for a single year
# keep the span narrow to start with: every extra year multiplies the number of records, and
# everything retrieved here is embedded in S2 and sent to the LLM in S3, which both cost time/money
YEAR_FROM = 2025
YEAR_TO   = 2025
# ...

# START OF SCRIPT

def main(
    KEY_PATH=KEY_PATH,    
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    STRINGS_FILE=STRINGS_FILE,
    YEAR_FROM=YEAR_FROM,
    YEAR_TO=YEAR_TO
):
    # import packages
    # imported inside main() rather than at the top of the file so that pipeline_publications.py
    # can import this script without immediately loading every dependency
    from dotenv import load_dotenv
    from datetime import datetime
    import dimcli
    from dimcli.utils import dsl_escape
    import pandas as pd
    import duckdb

    # 1. Load Dimensions API and search strings
    # Login to dimensions API requires a dsl.ini file stored on the computer
    # NOTE: that comment describes dimcli's other login option and no longer matches this code -
    # the key is read from the .env file at KEY_PATH (variable DIMENSIONS_API_KEY), no dsl.ini needed
    print("\nConnecting to the Dimensions API")
    
    load_dotenv(KEY_PATH) 
    dimcli.login(key=os.getenv("DIMENSIONS_API_KEY"))
    dsl = dimcli.Dsl()

    # Load search strings by reading the txt file
    with open(STRINGS_FILE, 'r') as f:
        search_strings = [line.strip() for line in f if line.strip()]

    # 2. Query dimensions
    print("\nStart query")

    # one query per search string; results are collected in this list and merged afterwards
    query = []
    for i in search_strings:
        # the txt file stores quotes escaped (\") so they survive copy-pasting from the Dimensions
        # web interface; undo that here because dsl_escape() below does the escaping properly
        query_string = i.replace('\\"', '\"')
        
        # only pull 100 publications per search term for testing
        # (the block below is a commented-out copy of the real query with 'limit 100' added; to try
        # a cheap test run, uncomment it and comment out the query_iterative call underneath)
        # query.append(dsl.query(f"""search publications in title_abstract_only for "{dsl_escape(query_string)}"
        #                     where year in [{YEAR_FROM}:{YEAR_TO}] 
        #                     return publications[id+title+abstract+year+type+authors+concepts_relevant+date+funders+
        #                     funder_countries+journal+open_access+research_org_names+research_org_countries+research_org_cities+times_cited]
        #                     limit 100"""))
        
        # query_iterative pages through the complete result set by itself (a single API call returns
        # at most 1000 records), so nothing has to be paged manually here.
        # "in title_abstract_only" = the search term must occur in the title or abstract, not in the
        # full text or keyword fields - that keeps the amount of loosely related noise manageable.
        # the field list after "return publications[...]" decides which metadata columns end up in
        # the database. If you change it, change S4_query_reverse.py the same way: both write into
        # the same CLASSIFICATION_TABLE and their columns have to match.
        query.append(dsl.query_iterative(f"""search publications in title_abstract_only for "{dsl_escape(query_string)}"
                            where year in [{YEAR_FROM}:{YEAR_TO}] 
                            return publications[id+title+abstract+year+type+authors+concepts_relevant+date+funders+
                            funder_countries+journal+open_access+research_org_names+research_org_countries+research_org_cities+times_cited]"""))

    # Convert to pandas dataframe and deduplicate by id
    # the search strings overlap on purpose, so the same publication is usually found several times
    query_df = pd.concat([q.as_dataframe() for q in query], ignore_index=True)
    print(f"\n{len(query_df)} publications retrieved from dimensions.")

    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)
    # records when this data was pulled from Dimensions (format yymmdd), so every row in the database
    # can later be traced to the day of retrieval - Dimensions data changes over time
    query_df['date_dimensions'] = datetime.today().strftime('%y%m%d')

    # 3. Filter articles
    # Filter for articles
    # only journal articles are kept: books, chapters, preprints, conference papers etc. are either
    # out of scope for the dataset or have abstracts too inconsistent for the classifiers
    query_df = query_df[query_df['type'] == 'article']

    print(f"\n{len(query_df)} publications remain after deduplication and filtering for articles.")

    # Filter publications that already are in the final database
    # publications classified in an earlier run are dropped here, so no publication is embedded or
    # sent to the LLM twice. This is why CLASSIFICATION_TABLE must never be cleaned out casually:
    # deleting rows there makes them come back as "new" in the next run.
    # Connect to SQL database
    db = duckdb.connect(database=DB_PATH)

    # on a completely fresh database CLASSIFICATION_TABLE does not exist yet, hence the check
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        n_before = len(query_df)
        query_df = query_df[~query_df['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(query_df)} rows already in {CLASSIFICATION_TABLE}.")

    # Reorder columns to match publications_classified if it exists
    # excludes columns computed later in the pipeline (S2_ML_classification.py, S3_LLM_scope.py, S6_LLM_labelling.py)
    # why this is needed: rows are later appended to CLASSIFICATION_TABLE positionally, so the run
    # table has to carry the same columns in the same order. reindex() also adds columns that
    # Dimensions did not return for this run (as empty ones) and drops unexpected extras.
    if CLASSIFICATION_TABLE in existing_tables:
        expected_cols = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                         if c not in ('pred_combined', 'pred_pillar',
                                      'scope_LLM', 'confidence_LLM', 'pillar_LLM',
                                      'plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
                                      'status_LLM', 'stop_reason_LLM',
                                      'date_ML', 'date_LLM', 'date_labelling')]
        query_df = query_df.reindex(columns=expected_cols)

    # 4. Add the queries to the database
    # Create run table and add queries
    print(f"\nStoring publications in database as {RUN_TABLE}")

    # CREATE OR REPLACE: re-running the same run overwrites its run table instead of appending, so a
    # repeated query never produces duplicates. Careful - it also means an existing table of that
    # name is lost, which is why every pipeline run uses a timestamped RUN_TABLE.
    # query_df is picked up directly from the local Python variable (a DuckDB feature).
    db.sql(f"CREATE OR REPLACE TABLE {RUN_TABLE} AS SELECT * FROM query_df")
    print(f"{len(query_df)} rows appended to {RUN_TABLE}.")

    # Close connection 
    db.close()

    print("Done!")


# only executed when the file is started directly (python S1_query_dimensions.py);
# importing it - as pipeline_publications.py does - does not run anything by itself
if __name__ == '__main__':
    main()
