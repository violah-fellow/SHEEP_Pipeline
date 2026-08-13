## STEP 1 of the patents pipeline
## Query script: query dimensions and store queried data in database
##
## What it does: searches Dimensions in two ways and merges the results - (a) by keyword, using the
## search strings in STRINGS_FILE, and (b) by CPC code, using the classification codes in
## CPC_SEARCH_FILE. Everything found is deduplicated, filtered by CPC, checked against what is
## already in the database, and stored as a new run table.
##
## Patent-specific concept: the FAMILY. The same invention is usually filed in several countries and
## in several versions - all of those share a family_id. The pipeline treats a family as one unit:
## deduplication, embedding and LLM calls all happen per family, not per document. That saves a large
## amount of API cost and keeps the dataset from counting the same invention many times over.
##
## Input:   dimensions_search_patents.txt (search strings), CPC_for_query.txt (CPC codes to search),
##          CPC_for_filter.txt (CPC codes to keep), DIMENSIONS_API_KEY from the .env file
## Output:  RUN_TABLE in patents.db - the raw input for S2_ML_classification.py
## All parameters that need to be changed are defined in the CONFIG section below.

import os

# CONFIG 
# edit parameters for this run here
# NOTE: started through pipeline_patents.py, these values are overridden by the pipeline's own
# CONFIG - for a pipeline run, edit them there and not here.

# Dimensions API
# path to API key
# the .env file holding DIMENSIONS_API_KEY; '../.env' is the one in the repository root
KEY_PATH = '../.env'

# Database
# path to DuckDB database
# a single file holding every table of this pipeline
DB_PATH = 'patents.db'
# table to store the new run data
# the pipeline names it run_<date>_<time>; an existing table of the same name is overwritten
RUN_TABLE = 'data_run_test'
# table for final classifications
# the long-lived table all runs accumulate into; used here to skip patents that are already known
CLASSIFICATION_TABLE = 'patents_classified'

# Queries
# path to txt file with search strings and CPC codes
# STRINGS_FILE    - one Dimensions DSL search string per line (keyword search)
# CPC_SEARCH_FILE - CPC codes that are searched directly, i.e. an extra way in for patents whose
#                   wording does not match any search string
# CPC_FILTER_FILE - the narrower list used AFTER the query: a patent is only kept if it carries at
#                   least one of these codes. This is the main noise filter of this step - the two
#                   CPC files serve different purposes and are deliberately not identical.
STRINGS_FILE = 'dimensions_search_patents.txt'
CPC_SEARCH_FILE = 'CPC_for_query.txt'
CPC_FILTER_FILE = 'CPC_for_filter.txt'

# Other parameters for search
# publication years to search, inclusive on both ends. Note this filters on publication_year - a
# patent family can appear in several years through later filings, which is what the version check
# further down deals with.
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
    # NOTE: that comment describes dimcli's other login option and no longer matches this code -
    # the key is read from the .env file at KEY_PATH (DIMENSIONS_API_KEY), no dsl.ini needed
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
    # two searches feed the same result list: one keyword query per search string (inside the loop)
    # and one CPC query (after the loop). Watch the indentation - the CPC query is intentionally
    # outside the loop and runs only once.
    print("\nStart query")

    query = []
    for i in search_strings:
        # the txt file stores quotes escaped (\") so they survive copy-pasting from the Dimensions
        # web interface; undo that here because dsl_escape() does the escaping properly
        query_string = i.replace('\\"', '\"')

        # only pull 100 patents per search term for testing
        # (everything commented out below is the test variant with 'limit 100'; to try a cheap run,
        # uncomment it and comment out the two query_iterative calls further down)

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
        # query_iterative pages through the full result set on its own. Unlike the publications
        # pipeline this searches the whole patent text, not just title/abstract - patent wording is
        # deliberately vague, so restricting the search would miss too much.
        # The field list must match the one in S4_query_reverse.py: both write into the same
        # CLASSIFICATION_TABLE and their columns have to line up.
        query.append(dsl.query_iterative(f"""search patents for "{dsl_escape(query_string)}"
                            where publication_year in [{YEAR_FROM}:{YEAR_TO}] 
                            return patents[id+family_id+application_number+title+abstract+cpc+jurisdiction+kind+year+priority_year+
                                publication_year+granted_year+filing_status+legal_status+inventor_names+original_assignee_names+current_assignee_names+
                                assignee_names+assignee_cities+assignee_countries+associated_grant_ids+funders+funder_countries+federal_support+
                                publications+researchers+times_cited+family_count] 
                    """))
        
    # CPC code search
    # second way in, independent of wording: every patent carrying one of the CPC codes from
    # CPC_for_query.txt. Runs once (not per search string) and its results are appended to the same
    # list; duplicates between the two searches are removed in the next step.
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
    # first pass: the same document found by both the keyword and the CPC search
    query_df = query_df.drop_duplicates(subset="id").reset_index(drop=True)

    # Remove version duplicates of the same patent
    # second pass: the same invention published several times in the same country - e.g. first as an
    # application, later as a granted patent. Grouping by (family_id, jurisdiction, priority_year)
    # and keeping the first row after sorting by publication_year and kind descending keeps the
    # newest and highest-value version per country. Versions in OTHER countries are kept - they are
    # separate legal documents, and the family logic in S2/S3 handles them.
    query_df = query_df.sort_values(['publication_year', 'kind'], ascending=[False, False]).groupby(["family_id", "jurisdiction", "priority_year"]).head(1)
    print(f"\n{len(query_df)} patents remain after deduplication.")

    # clean abstract
    # patent abstracts from Dimensions often contain HTML tags - they would end up in the embedding
    # text and in the LLM prompt, so they are stripped here
    query_df['abstract'] = query_df['abstract'].str.replace(r'<[^>]*>', '', regex=True)
    # date of retrieval (yymmdd), so every row can be traced to when it was pulled
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

    # 3. Filter by CPC codes
    # Get CPC codes for filtering
    # Load CPC codes for filtering by reading the txt file
    with open(CPC_FILTER_FILE, 'r') as f:
        cpc_filter = f.read().splitlines()

    # keep a patent if at least one of its CPC codes is in the filter list. Note the two conditions
    # in front of the any(): patents WITHOUT CPC information are kept deliberately - a missing code
    # is not evidence of being off-topic, and the ML/LLM steps can still sort them out.
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
        # this is the patent-specific part of the deduplication and the only place in the pipeline
        # where rows are DELETED from CLASSIFICATION_TABLE: if the current query returns a newer
        # version of a document that is already in the dataset (e.g. the granted patent for an
        # application classified earlier), the old row is removed and replaced by the new one.
        # Conversely, an older or equally old version in the query is dropped.
        # Consequence worth knowing: the classifications of the deleted row are lost, and the new
        # version goes through the ML and LLM steps again.
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


# only executed when the file is started directly (python S1_query_dimensions.py);
# importing it - as pipeline_patents.py does - does not run anything by itself
if __name__ == '__main__':
    main()
