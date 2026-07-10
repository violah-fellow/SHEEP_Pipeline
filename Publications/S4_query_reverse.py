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
DB_PATH = 'publications.db'
# table where run queries were stored
RUN_TABLE = 'data_run_test'
REVERSE_TABLE = RUN_TABLE + '_reverse'
# table for final classifications
CLASSIFICATION_TABLE = 'publications_classified'

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

    # 2. Extract top researcher ID's
    # Connect to SQL database
    db = duckdb.connect(database=DB_PATH)

    # Retrieve run table and convert to pandas dataframe
    data = db.sql(f"SELECT * FROM {RUN_TABLE}").df()

    # Filter for in scope publications and retrieve publication ID's
    data = data[data['pred_combined'] == 1] 
    pub_ids = data['id'].tolist()

    # Function to batch publication ID's    
    def chunks(list, n):
        for i in range(0, len(list), n):
            yield list[i:i + n]

    # Get researcher ID's
    print("Retrieve top researcher ID's from dimensions")

    all_researchers = []

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
    top_researchers = (
        all_researchers
        .groupby(["id", "first_name", "last_name"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .reset_index(drop=True)
    )

    researcher_ids = json.dumps(top_researchers.head(500)["id"].tolist())

    # 3. Query dimensions with researcher ID's
    print("\nStart reverse query")

    # only pull 100 publications per search term for testing
    # query = dsl.query(f"""search publications
    #                     where researchers in {researcher_ids}
    #                     and year={YEAR} 
    #                     return publications[id+title+abstract+year+type+authors+concepts_relevant+date+funders+
    #                         funder_countries+journal+open_access+research_org_names+research_org_countries+research_org_cities+times_cited]
    #                     limit 10""")
    
    query = dsl.query_iterative(f"""search publications 
                      where researchers.id in {researcher_ids} 
                      and year={YEAR} 
                      return publications[id+title+abstract+year+type+authors+concepts_relevant+date+funders+
                      funder_countries+journal+open_access+research_org_names+research_org_countries+research_org_cities+times_cited]""")

    # Convert to pandas dataframe and deduplicate by id
    query_df = query.as_dataframe()
    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)
    query_df['date_dimensions'] = datetime.today().strftime('%y%m%d')

    # 3. Filter articles
    # Filter for articles
    query_df = query_df[query_df['type'] == 'article']

    # Filter publications that already are in the final database
    # Connect to SQL database
    db = duckdb.connect(database=DB_PATH)

    existing_tables = db.sql("SHOW TABLES").df()['name'].tolist()
    if CLASSIFICATION_TABLE in existing_tables:
        existing_ids = db.sql(f"SELECT id FROM {CLASSIFICATION_TABLE}").df()['id']
        n_before = len(query_df)
        query_df = query_df[~query_df['id'].isin(existing_ids)].reset_index(drop=True)
        print(f"{n_before - len(query_df)} rows already in {CLASSIFICATION_TABLE}.")

    # Reorder columns to match publications_classified
    # excludes columns computed later in the pipeline (S2_ML_classification.py, S3_LLM_scope.py, S6_LLM_labelling.py)
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


if __name__ == '__main__':
    main()
