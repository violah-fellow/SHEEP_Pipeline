## Query script: query dimensions and store queried data in database
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
RUN_TABLE = 'data_run_test'
REVERSE_TABLE = RUN_TABLE + '_reverse'
# table for final classifications
CLASSIFICATION_TABLE = 'patents_classified'

# Queries
# Other parameters for search
YEAR = 2025
# ...

# START OF SCRIPT

def main(
    KEY_PATH,
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    REVERSE_TABLE=REVERSE_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    YEAR=YEAR,
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
    run_data = run_data[run_data['pred_combined'] == 1]
    family_ids = run_data['family_id'].tolist()

    # Function to batch family ID's    
    def chunks(list, n):
        for i in range(0, len(list), n):
            yield list[i:i + n]

    # 3. Query dimensions with family ID's
    print("\nStart reverse query")

    # only pull 100 publications per search term for testing
    # query = []
    # for batch in chunks(family_ids, 500):
    #     q = dsl.query(f"""search patents
    #       where family_id in {json.dumps(batch)}
    #       return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
    #                     publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
    #                     assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
    #                     publications+researchers+times_cited+family_count]
    #       limit 100 """)
    #     query.append(q)

    # full query
    query = []
    for batch in chunks(family_ids, 500):
        q = dsl.query_iterative(f"""search patents
        where family_id in {json.dumps(batch)}
        return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
                        publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
                        assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
                        publications+researchers+times_cited+family_count]
        """)
        query.append(q)

    # Convert to pandas dataframe and deduplicate by id    
    query_df = pd.concat([q.as_dataframe() for q in query], ignore_index=True)
    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)

    # Remove version duplicates of the same patent
    query_df = query_df.sort_values(['publication_year', 'kind'], ascending=[False, False]).groupby(["family_id", "jurisdiction", "application_number"]).head(1)

    # clean abstract
    query_df['abstract'] = query_df['abstract'].str.replace(r'<[^>]*>', '', regex=True)
    query_df['date_dimensions'] = datetime.today().strftime('%y%m%d')

    # Filter publications that already are in the final database
    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        n_before = len(query_df)
        query_df = query_df[~query_df['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(query_df)} rows already in {CLASSIFICATION_TABLE}.")

        # Check for newer versions of already-classified patents
        # (same family/jurisdiction/application combo but higher publication_year)
        existing_versions = db.sql(f"""
            SELECT family_id, jurisdiction, application_number, publication_year AS publication_year_existing, id AS id_existing
            FROM {CLASSIFICATION_TABLE}
        """).df()

        version_matches = query_df.merge(
            existing_versions, on=['family_id', 'jurisdiction', 'application_number'], how='inner'
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
    data_classified = data.reindex(columns=output_columns).copy()
    data_classified['pred_combined'] = data_classified['pred_combined'].map({1: 'in', 0: 'out'})
    db.register('data_classified', data_classified)

    cols_str = ", ".join(f'"{c}"' for c in output_columns)
    db.sql(f"INSERT INTO {CLASSIFICATION_TABLE} ({cols_str}) SELECT * FROM data_classified")
    
    print(f"{len(data_classified)} rows appended to {CLASSIFICATION_TABLE}.")

    # Close connection 
    db.close()

    print("Done!")


if __name__ == '__main__':
    main()
