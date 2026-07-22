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
# table to store the new run data
RUN_TABLE = 'data_run_test'
# table for final classifications
CLASSIFICATION_TABLE = 'patents_classified'

# Queries
# path to txt file with search strings and CPC codes
STRINGS_FILE = 'dimensions_search_patents.txt'
CPC_SEARCH_FILE = 'CPC_for_query.txt'
CPC_FILTER_FILE = 'CPC_for_filter.txt'

# Other parameters for search
YEAR_FROM = 2023
YEAR_TO   = 2024
# ...

# START OF SCRIPT

def main(
    KEY_PATH=KEY_PATH,    
    DB_PATH=DB_PATH,
    RUN_TABLE=RUN_TABLE,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    STRINGS_FILE=STRINGS_FILE,
    CPC_SEARCH_FILE=CPC_SEARCH_FILE,
    CPC_FILTER_FILE=CPC_FILTER_FILE,
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
    print("\nConnecting to the Dimensions API")
    
    load_dotenv(KEY_PATH) 
    dimcli.login(key=os.getenv("DIMENSIONS_API_KEY"))
    dsl = dimcli.Dsl()

    # Load search strings by reading the txt file
    with open(STRINGS_FILE, 'r') as f:
        search_strings = [line.strip() for line in f if line.strip()]

    # Load CPC codes for querying by reading the txt file
    with open(CPC_SEARCH_FILE, 'r') as f:
        cpc_search = f.read().splitlines()

    # 2. Query dimensions
    print("\nStart query")

    query = []
    for i in search_strings:
        query_string = i.replace('\\"', '\"')
        
        # only pull 100 patents per search term for testing

    #     # string search
    #     query.append(dsl.query(f"""search patents for "{dsl_escape(query_string)}"
    #                            where publication_year in [{YEAR_FROM}:{YEAR_TO}] 
    #                            return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
    #                             publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
    #                             assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
    #                             publications+researchers+times_cited+family_count] 
    #                            limit 100"""))
        
    # # CPC code search
    # query.append(dsl.query(f"""search patents 
    #                         where cpc in {json.dumps(cpc_search)}
    #                         and publication_year in [{YEAR_FROM}:{YEAR_TO}] 
    #                         return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
    #                         publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
    #                         assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
    #                         publications+researchers+times_cited+family_count]
    #                         limit 100"""))
        
        # full search

        # string search
        query.append(dsl.query_iterative(f"""search patents for "{dsl_escape(query_string)}"
                            where publication_year in [{YEAR_FROM}:{YEAR_TO}] 
                            return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
                                publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
                                assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
                                publications+researchers+times_cited+family_count] 
                    """))
        
    # CPC code search
    query.append(dsl.query_iterative(f"""search patents 
                        where cpc in {json.dumps(cpc_search)}
                        and publication_year in [{YEAR_FROM}:{YEAR_TO}]  
                        return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
                            publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
                            assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
                            publications+researchers+times_cited+family_count]
                """))

    # Convert to pandas dataframe  
    query_df = pd.concat([q.as_dataframe() for q in query], ignore_index=True)
    print(f"\n{len(query_df)} patents retrieved from dimensions.")

    # deduplicate by id
    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)

    # Remove version duplicates of the same patent
    query_df = query_df.sort_values(['publication_year', 'kind'], ascending=[False, False]).groupby(["family_id", "jurisdiction", "priority_year"]).head(1)
    print(f"\n{len(query_df)} patents remain after deduplication.")

    # clean abstract
    query_df['abstract'] = query_df['abstract'].str.replace(r'<[^>]*>', '', regex=True)
    query_df['date_dimensions'] = datetime.today().strftime('%y%m%d')

    # 3. Filter by CPC codes
    # Get CPC codes for filtering
    # Load CPC codes for filtering by reading the txt file
    with open(CPC_FILTER_FILE, 'r') as f:
        cpc_filter = f.read().splitlines()

    cpc_mask = query_df['cpc'].apply(
        lambda codes: not isinstance(codes, list) or not codes or any(c in cpc_filter for c in codes)
    )
    
    query_df = query_df[cpc_mask]
    print(f"\n{len(query_df)} patents remain after filtering by CPC codes.")

    # Filter patents that already are in the final database
    # Connect to SQL database
    db = duckdb.connect(database=DB_PATH)

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

    # Reorder columns to match patents_classified if it exists
    # excludes columns computed later in the pipeline (S2_ML_classification.py, S3_LLM_scope.py, S7_LLM_labelling.py)
    if CLASSIFICATION_TABLE in existing_tables:
        expected_cols = [c for c in db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE} LIMIT 0").df().columns.tolist()
                         if c not in ('pred_combined', 'pred_pillar', 'proba_scope',
                                      'scope_LLM', 'confidence_LLM', 'pillar_LLM',
                                      'plant_based_LLM', 'fermentation_LLM', 'cultivated_LLM',
                                      'cross_cutting_LLM', 'status_LLM', 'stop_reason_LLM',
                                      'date_ML', 'date_LLM', 'date_labelling')]
        query_df = query_df.reindex(columns=expected_cols)

    # 4. Add the queries to the database
    # Create run table and add queries
    print(f"\nStoring patents in database as {RUN_TABLE}")

    db.sql(f"CREATE OR REPLACE TABLE {RUN_TABLE} AS SELECT * FROM query_df")
    print(f"{len(query_df)} rows appended to {RUN_TABLE}.")

    # Close connection 
    db.close()

    print("Done!")


if __name__ == '__main__':
    main()
