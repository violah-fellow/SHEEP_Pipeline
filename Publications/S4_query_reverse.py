## STEP 4 of the publications pipeline
## Query script: query dimensions and store queried data in database
## (reverse query - i.e. searching via people instead of via search terms)
##
## Idea behind this step: a keyword search only finds publications that use the expected wording.
## So the pipeline turns the search around - it looks up the researchers who wrote the in-scope
## publications of this run, and then fetches ALL of their publications from the year range. That
## also catches relevant work whose title and abstract never match one of the search strings.
##
## Input:   RUN_TABLE with the ML predictions from S2_ML_classification.py
## Output:  REVERSE_TABLE ('<RUN_TABLE>_reverse'), which pipeline_publications.py then sends through
##          the same S2 -> S3 steps as the normal run table
##
## Note that the additional records are not free: everything found here is embedded and sent to the
## LLM as well, so this step usually dominates the cost of a run.
## All parameters that need to be changed are defined in the CONFIG section below.

import os

# CONFIG 
# edit parameters for this run here

# Dimensions API
# path to API key
KEY_PATH = '../.env'

# Database
# path to DuckDB database
DB_PATH = 'publications.db'
# table where run queries were stored
# the classified run table of this run - the researchers are taken from its in-scope publications
RUN_TABLE = 'data_run_test'
# output table; the pipeline uses the same naming convention ('<run table>_reverse')
REVERSE_TABLE = RUN_TABLE + '_reverse'
# table for final classifications
# used here only to skip publications that are already in the dataset
CLASSIFICATION_TABLE = 'publications_classified'

# Queries
# Other parameters for search
# use the same year range as in S1_query_dimensions.py - otherwise the reverse search covers a
# different period than the keyword search and the dataset becomes inconsistent
YEAR_FROM = 2025
YEAR_TO   = 2025
# ...

# START OF SCRIPT

def main(
    KEY_PATH=KEY_PATH,
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    REVERSE_TABLE=REVERSE_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    YEAR_FROM=YEAR_FROM,
    YEAR_TO=YEAR_TO
):
    # import packages
    from dotenv import load_dotenv
    from datetime import datetime
    import dimcli
    from dimcli.utils import dsl_escape
    import pandas as pd
    import duckdb
    import json

    # 1. Load Dimensions API and search strings
    # Login to dimensions API requires a dsl.ini file stored on the computer
    # NOTE: as in S1, that comment no longer matches the code - the key comes from the .env file at
    # KEY_PATH (DIMENSIONS_API_KEY). No search strings are read here either; this step searches via
    # researcher IDs instead of keywords.
    print("\nConnecting to the Dimensions API")
    
    load_dotenv(KEY_PATH) 
    dimcli.login(key=os.getenv("DIMENSIONS_API_KEY"))
    dsl = dimcli.Dsl()

    # 2. Extract top researcher ID's
    # Connect to SQL database
    db = duckdb.connect(database=DB_PATH)

    # Retrieve run table and convert to pandas dataframe
    data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()

    # Filter for in scope publications and retrieve publication ID's
    # deliberately the ML prediction (pred_combined == 1) and not the curated scope: this step runs
    # before the manual review, so only the machine decision is available at this point
    data = data[data['pred_combined'] == 1]
    pub_ids = data['id'].tolist()

    # Function to batch publication ID's
    # helper that splits a long list into pieces of n items
    def chunks(list, n):
        for i in range(0, len(list), n):
            yield list[i:i + n]

    # Get researcher ID's
    print("Retrieve top researcher ID's from dimensions")

    all_researchers = []

    # in batches of 200 IDs because a single DSL query can only take a limited number of values;
    # 'limit 1000' caps the number of researchers returned per batch
    for batch in chunks(pub_ids, 200):
        ids_str = json.dumps(batch)  
        result = dsl.query(f"""
            search publications
            where id in {ids_str}
            return researchers 
            limit 1000
        """)
        df = result.as_dataframe()
        if df is not None and not df.empty:
            all_researchers.append(df)

    all_researchers = pd.concat(all_researchers, ignore_index=True)

    # Sum counts across batches
    # 'count' = how many of the in-scope publications of this run belong to that researcher. A
    # researcher appearing in several batches is added up here, so the ranking below is based on the
    # total number of hits, not on the batch a researcher happened to appear in.
    top_researchers = (
        all_researchers
        .groupby(["id", "first_name", "last_name"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    # only the 500 most productive researchers are followed up. This cut-off is the main cost control
    # of this step - raising it increases the number of retrieved publications considerably, and each
    # of them is embedded and sent to the LLM afterwards. The value is hardcoded on purpose but is a
    # sensible thing to tune (it is not exposed in the CONFIG block).
    researcher_ids = json.dumps(top_researchers.head(500)["id"].tolist())

    # 3. Query dimensions with researcher ID's
    print("\nStart reverse query")

    # only pull 100 publications per search term for testing
    # (commented-out test variant with 'limit 10'; swap it with the query_iterative call below for a
    # quick trial run - note that no scope filter is applied here, this searches by researcher, so a
    # full run can return a large number of records)
    # query = dsl.query(f"""search publications
    #                     where researchers in {researcher_ids}
    #                     and year in [{YEAR_FROM}:{YEAR_TO}]  
    #                     return publications[id+title+abstract+year+type+authors+concepts_relevant+date+funders+
    #                         funder_countries+journal+open_access+research_org_names+research_org_countries+research_org_cities+times_cited]
    #                     limit 10""")
    
    # the returned field list must match the one in S1_query_dimensions.py - both tables end up in
    # the same CLASSIFICATION_TABLE, so their columns have to be identical
    query = dsl.query_iterative(f"""search publications 
                      where researchers.id in {researcher_ids} 
                      and year in [{YEAR_FROM}:{YEAR_TO}]  
                      return publications[id+title+abstract+year+type+authors+concepts_relevant+date+funders+
                      funder_countries+journal+open_access+research_org_names+research_org_countries+research_org_cities+times_cited]""")

    # Convert to pandas dataframe and deduplicate by id
    query_df = query.as_dataframe()
    print(f"\n{len(query_df)} publications retrieved from dimensions.")

    # Deduplicate
    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)
    query_df['date_dimensions'] = datetime.today().strftime('%y%m%d')

    # 3. Filter articles
    # (numbering slip - this is the second block labelled 3.)
    # Filter for articles
    # same restriction as in S1: journal articles only
    query_df = query_df[query_df['type'] == 'article']

    print(f"\n{len(query_df)} publications remain after deduplication and filtering for articles.")

    # Filter publications that already are in the final database
    # this is the important filter of this step: a reverse search returns a lot of publications the
    # keyword search already found, plus everything from earlier runs. Only genuinely new records
    # are embedded and classified.
    # Connect to SQL database
    # (the connection is already open from step 2 - connecting again is redundant but harmless)
    db = duckdb.connect(database=DB_PATH)

    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        n_before = len(query_df)
        query_df = query_df[~query_df['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(query_df)} rows already in {CLASSIFICATION_TABLE}.")

    # Reorder columns to match publications_classified
    # excludes columns computed later in the pipeline (S2_ML_classification.py, S3_LLM_scope.py, S6_LLM_labelling.py)
    # unlike S1 there is no "if the table exists" check here: this step always runs after S2, so
    # CLASSIFICATION_TABLE does exist by now
    expected_cols = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                     if c not in ('pred_combined', 'pred_pillar',
                                  'scope_LLM', 'confidence_LLM', 'pillar_LLM',
                                  'plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM', 'cross_cutting_LLM',
                                  'status_LLM', 'stop_reason_LLM',
                                  'date_ML', 'date_LLM', 'date_labelling')]
    query_df = query_df.reindex(columns=expected_cols)

    # 4. Add the queries to the database
    # Create reverse run table and add queries
    print(f"\nStoring publications in database as {REVERSE_TABLE}")

    db.sql(f"CREATE OR REPLACE TABLE {REVERSE_TABLE} AS SELECT * FROM query_df")
    print(f"{len(query_df)} rows appended to {REVERSE_TABLE}.")

    # Close connection 
    db.close()

    print("Done!")


# only executed when the file is started directly - note the KEY_PATH caveat in main() above:
# main() without arguments raises a TypeError, use main(KEY_PATH='../.env')
if __name__ == '__main__':
    main()
