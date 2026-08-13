## STEP 4 of the patents pipeline
## Query script: query dimensions and store queried data in database
## (reverse query - searching by patent family instead of by keyword)
##
## Idea behind this step: if one document of a family is in scope, its siblings in other countries
## are too - they describe the same invention. So this step takes the family IDs of the in-scope
## patents of this run and fetches every remaining family member from Dimensions.
##
## Important difference from the publications pipeline: these patents are NOT classified again.
## They inherit the ML and LLM results of their family (step 5) and are written straight into
## CLASSIFICATION_TABLE (step 6) - so no embedding, no API call, no cost. It also means the pipeline
## ends after this step plus the assignee clean-up; nothing here needs S2 or S3 again.
##
## Input:   RUN_TABLE with the ML and LLM columns from S2/S3
## Output:  REVERSE_TABLE ('<RUN_TABLE>_reverse') and the same rows appended to CLASSIFICATION_TABLE
## All parameters that need to be changed are defined in the CONFIG section below.

import os

# CONFIG 
# edit parameters for this run here

# Dimensions API
# path to API key
KEY_PATH = '../.env'

# Database
# path to DuckDB database
DB_PATH = 'patents.db'
# table where run queries were stored
# the classified run table of this run - family IDs are taken from its in-scope patents
RUN_TABLE = 'run_test'
# output table; the pipeline uses the same naming convention ('<run table>_reverse')
REVERSE_TABLE = RUN_TABLE + '_reverse'
# table for final classifications
CLASSIFICATION_TABLE = 'patents_classified'

# Queries
# Other parameters for search
# use the same year range as in S1_query_dimensions.py. Note this limits which family members are
# found at all: siblings published outside the range stay invisible to the pipeline.
YEAR_FROM = 2023
YEAR_TO   = 2024
# ...

# START OF SCRIPT

def main(
    KEY_PATH=KEY_PATH,
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    REVERSE_TABLE=REVERSE_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    YEAR_FROM=YEAR_FROM,
    YEAR_TO=YEAR_TO,
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
    # KEY_PATH. No search strings are read here either; this step searches by family ID.
    print("\nConnecting to the Dimensions API")
    
    load_dotenv(KEY_PATH) 
    dimcli.login(key=os.getenv("DIMENSIONS_API_KEY"))
    dsl = dimcli.Dsl()

    # 2. Extract family ID's
    # Connect to SQL database
    db = duckdb.connect(database=DB_PATH)

    # Retrieve run table and convert to pandas dataframe
    run_data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()

    # Filter for in scope patents and retrieve family ID's
    # the ML decision (0/1 in the run table), not the curated one - the manual review only happens
    # later, in S6. run_data is reused further down as the source of the inherited predictions.
    run_data = run_data[run_data['pred_combined'] == 1]
    family_ids = run_data['family_id'].dropna().astype(int).tolist()

    # Function to batch family ID's
    # helper that splits a long list into pieces of n items
    def chunks(list, n):
        for i in range(0, len(list), n):
            yield list[i:i + n]

    # 3. Query dimensions with family ID's
    print("\nStart reverse query")

    # only pull 100 publications per search term for testing
    # query = []
    # for batch in chunks(family_ids, 500):
    #     q = dsl.query(f"""search patents
    #       where family_id in {json.dumps(batch)} and publication_year in [{YEAR_FROM}:{YEAR_TO}] 
    #       return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
    #                     publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
    #                     assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
    #                     publications+researchers+times_cited+family_count]
    #       limit 100 """)
    #     query.append(q)

    # full query
    # in batches of 500 family IDs, because a single DSL query can only take a limited number of
    # values; the field list must match S1_query_dimensions.py
    query = []
    for batch in chunks(family_ids, 500):
        q = dsl.query_iterative(f"""search patents
        where family_id in {json.dumps(batch)} and publication_year in [{YEAR_FROM}:{YEAR_TO}] 
        return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
                        publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
                        assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
                        publications+researchers+times_cited+family_count]
        """)
        query.append(q)

    # Convert to pandas dataframe    
    query_df = pd.concat([q.as_dataframe() for q in query], ignore_index=True)
    print(f"\n{len(query_df)} patents retrieved from dimensions.")

    # deduplicate by id
    # same two-stage deduplication as in S1: first identical documents, ...
    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)

    # Remove version duplicates of the same patent
    # ... then several versions of the same document in the same country - keep the newest and
    # highest-value one per (family_id, jurisdiction, priority_year)
    query_df = query_df.sort_values(['publication_year', 'kind'], ascending=[False, False]).groupby(["family_id", "jurisdiction", "priority_year"]).head(1)
    print(f"\n{len(query_df)} patents remain after deduplication.")

    # clean abstract
    # strip HTML tags that Dimensions returns inside patent abstracts
    query_df['abstract'] = query_df['abstract'].str.replace(r'<[^>]*>', '', regex=True)
    query_df['date_dimensions'] = datetime.today().strftime('%y%m%d')

    # Dimensions returns a different subset of fields per entry within these list-of-dict
    # columns (e.g. some funders lack 'state_name'), which otherwise makes DuckDB's struct-type
    # inference fail outright when creating the table. Fill every dict to the same key set.
    def _normalize_struct_list_column(series):
        all_keys = set()
        for lst in series:
            if isinstance(lst, list):
                for d in lst:
                    if isinstance(d, dict):
                        all_keys.update(d.keys())
        if not all_keys:
            return series
        def _fill(lst):
            if not isinstance(lst, list):
                return lst
            return [{k: d.get(k) for k in all_keys} for d in lst if isinstance(d, dict)]
        return series.apply(_fill)

    for _c in ('assignee_cities', 'assignee_countries', 'funder_countries', 'funders'):
        if _c in query_df.columns:
            query_df[_c] = _normalize_struct_list_column(query_df[_c])

    # Filter publications that already are in the final database
    # the crucial filter of this step: a family query returns above all the patents that are already
    # in the dataset (that is where the family IDs came from). Only genuinely new siblings remain.
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        n_before = len(query_df)
        query_df = query_df[~query_df['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(query_df)} rows already in {CLASSIFICATION_TABLE}.")

        # Check for newer versions of already-classified patents
        # (same family/jurisdiction/priority_year combo but higher publication_year)
        existing_versions = db.sql(f"""
            SELECT family_id, jurisdiction, priority_year, publication_year AS publication_year_existing, id AS id_existing
            FROM {CLASSIFICATION_TABLE}
        """).df()

        version_matches = query_df.merge(
            existing_versions, on=['family_id', 'jurisdiction', 'priority_year'], how='inner'
        )

        newer = version_matches[version_matches['publication_year'] > version_matches['publication_year_existing']]
        older_or_same = version_matches[version_matches['publication_year'] <= version_matches['publication_year_existing']]

        if not newer.empty:
            db.register('superseded', pd.DataFrame({'id': newer['id_existing'].tolist()}))
            db.sql(f"DELETE FROM {CLASSIFICATION_TABLE} WHERE id IN (SELECT id FROM superseded)")
            print(f"{len(newer)} superseded rows deleted from {CLASSIFICATION_TABLE}.")

        if not older_or_same.empty:
            query_df = query_df[~query_df['id'].isin(older_or_same['id'])].reset_index(drop=True)
            print(f"{len(older_or_same)} older/same-version rows dropped.")

        # Reorder columns to match patents_classified
        # excludes columns computed later in the pipeline (S2_ML_classification.py, S3_LLM_scope.py, S7_LLM_labelling.py)
        expected_cols = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                         if c not in ('pred_combined', 'pred_pillar', 'proba_scope',
                                      'scope_LLM', 'confidence_LLM', 'pillar_LLM',
                                      'plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM',
                                      'cross_cutting_LLM', 'status_LLM', 'stop_reason_LLM',
                                      'date_ML', 'date_LLM', 'date_labelling')]
        query_df = query_df.reindex(columns=expected_cols)

    # 4. Add the queries to the database
    # Create reverse run table and add queries
    print(f"\nStoring publications in database as {REVERSE_TABLE}")

    db.sql(f"CREATE OR REPLACE TABLE {REVERSE_TABLE} AS SELECT * FROM query_df")
    print(f"{len(query_df)} rows appended to {REVERSE_TABLE}.")

    # 5. Add scope and pillar information from patents of the same family
    # the heart of this step: instead of classifying the new patents, they inherit ALL prediction
    # columns - the ML ones and the LLM ones - from their already scored family sibling, joined on
    # family_id. That is why this list is much longer than the corresponding one in S2.
    print(f"\nUpdating '{REVERSE_TABLE}' in database with prediction columns.")

    new_columns = {
        'proba_scope':       'DOUBLE',
        'pred_scope':        'INTEGER',
        'threshold_scope':   'DOUBLE',
        'proba_pillar':      'DOUBLE',
        'pred_pillar':       'VARCHAR',
        'pred_combined':     'INTEGER',
        'date_ML':           'VARCHAR',
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
    for col, dtype in new_columns.items():
        db.sql(f"ALTER TABLE {REVERSE_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype}")

    # make sure there is only one patent per family ID
    # required for the join below - otherwise a family with several scored members would produce
    # duplicate matches. Which member is kept does not matter: they all carry the same predictions.
    run_data = run_data.drop_duplicates(subset="family_id").reset_index(drop=True)

    available_cols = ['family_id'] + [c for c in new_columns.keys() if c in run_data.columns]
    db.register('data', run_data[available_cols])
    set_clause = ", ".join(f"{c} = data.{c}" for c in new_columns if c in run_data.columns)
    db.sql(f"""
        UPDATE {REVERSE_TABLE}
        SET {set_clause}
        FROM data
        WHERE {REVERSE_TABLE}.family_id = data.family_id
    """)

    # 6. Append to patents_classified
    print(f"\nAppending to {CLASSIFICATION_TABLE} table.")

    # get output columns for CLASSIFICATION_TABLE; exclude date_labelling (set by S7, not yet run)
    output_columns = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                      if c != 'date_labelling']
    
    # get data with predictions from reverse_table
    data = db.sql(f"SELECT * FROM {REVERSE_TABLE}").df()

    # convert prediction in / out and add to CLASSIFICATION_TABLE
    # 0/1 in the run tables, 'in'/'out' in CLASSIFICATION_TABLE - same convention as in S2
    data_classified = data.reindex(columns=output_columns).copy()
    data_classified['pred_combined'] = data_classified['pred_combined'].map({1: 'in', 0: 'out'})
    db.register('data_classified', data_classified)

    cols_str = ", ".join(f'"{c}"' for c in output_columns)
    db.sql(f"INSERT INTO {CLASSIFICATION_TABLE} ({cols_str}) SELECT * FROM data_classified")
    
    print(f"{len(data_classified)} rows appended to {CLASSIFICATION_TABLE}.")

    # Close connection
    db.close()

    print("Done!")


# only executed when the file is started directly - note the KEY_PATH caveat in main() above:
# main() without arguments raises a TypeError, use main(KEY_PATH='../.env')
if __name__ == '__main__':
    main()
