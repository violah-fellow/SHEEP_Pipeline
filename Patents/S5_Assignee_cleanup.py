## Clean-up script for company names: match with Lucas' company list, convert to upper case and match with company database
## Can be run standalone (uses CONFIG defaults) or imported and called as main().

# CONFIG
# edit parameters for this run here

# Database
# path to DuckDB database
DB_PATH = 'patents.db'
# table with final classifications
CLASSIFICATION_TABLE = 'patents_classified'

# Column with assignee names
ASSIGNEE_COLUMN = 'assignee_names'
# Column for cleaned assignee names
CLEAN_COLUMN = 'assignee_names_cleaned'

# Path to company list with varying names matched to one true company names
COMPANY_LIST = 'company_list.csv'
# Path to company database with known companies for rapidfuzz matching
COMPANY_DATABASE = 'company_database.csv'
# IMPORTANT: Column names in company list and company database are hard-coded!
# If they are ever changed, the code needs to be adjusted as well!
# Leaving this for now because they are resources that should only be updated but not changed!
# Column names:
# Company list: 'Name' = names as they might appear in assignee names, 'Cleaned-up Name' = uniform, cleaned company names
# Company database: 'Company' = names of companies (match the cleaned-up names in company list)

# START OF SCRIPT

def main(
    DB_PATH=DB_PATH,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    ASSIGNEE_COLUMN=ASSIGNEE_COLUMN,
    CLEAN_COLUMN=CLEAN_COLUMN,
    COMPANY_LIST=COMPANY_LIST,
    COMPANY_DATABASE=COMPANY_DATABASE,
):
    import duckdb
    import numpy as np
    import pandas as pd
    import unicodedata
    from rapidfuzz import process, fuzz, utils

    # 1. Define functions for clean-up

    # Function to map the different versions of company names to the cleaned-up versions
    # Returns the mapped value for a single cell (string or list of strings)
    def map_companies(names, name_map):
        if names is None or (not isinstance(names, (list, np.ndarray)) and pd.isna(names)):
            return names
        if isinstance(names, (list, np.ndarray)):
            return [name_map.get(n, n).upper() for n in names]
        return name_map.get(names, names).upper()

    # Function to normalise text by removing accents and other special characters
    def normalize(s):
        return unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode('ascii')

    # Function to match slightly different versions of company names to companies in database
    # Returns the matched value for a single cell (string or list of strings)
    def match_company(names, companies_normalized, companies, cutoff=95):

        def get_match(n):
            result = process.extractOne(
                normalize(n), companies_normalized,
                scorer=fuzz.WRatio,
                processor=utils.default_process,
                score_cutoff=cutoff
            )
            if result:
                match, score, idx = result
                original_match = companies[idx]
                if len(original_match) >= len(n) * 0.5:
                    return original_match
            return n

        if names is None or (not isinstance(names, (list, np.ndarray)) and pd.isna(names)):
            return names
        if isinstance(names, (list, np.ndarray)):
            return [get_match(n) for n in names]
        return get_match(names)

    # 2. Retrieve data from CLASSIFICATION_TABLE, load company data

    db = duckdb.connect(database=DB_PATH)
    data = db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE}").df()

    company_list = pd.read_csv(COMPANY_LIST)
    company_database = pd.read_csv(COMPANY_DATABASE)

    # Build name map and combined company list
    name_map = dict(zip(company_list['Name'], company_list['Cleaned-up Name']))
    companies = pd.concat([company_list['Cleaned-up Name'], company_database['Company']]).drop_duplicates().tolist()

    # Precompute normalised company names once
    companies_normalized = [normalize(c) for c in companies]

    # 3. Do the clean-up

    # Map known name variants to cleaned-up names and convert to upper case
    data['tmp'] = data[ASSIGNEE_COLUMN].apply(lambda x: map_companies(x, name_map))

    # Fuzzy-match remaining names against company database
    data[CLEAN_COLUMN] = data['tmp'].apply(lambda x: match_company(x, companies_normalized, companies))

    data = data.drop(columns='tmp')
    # ensure result is always a list so DuckDB can cast to VARCHAR[]
    data[CLEAN_COLUMN] = data[CLEAN_COLUMN].apply(
        lambda x: list(x) if isinstance(x, (list, np.ndarray)) else ([x] if isinstance(x, str) else x)
    )

    # 4. Add the cleaned-up column to CLASSIFICATION_TABLE

    db.sql(f"ALTER TABLE {CLASSIFICATION_TABLE} ADD COLUMN IF NOT EXISTS {CLEAN_COLUMN} VARCHAR[]")

    db.register('data_clean', data[['id', CLEAN_COLUMN]])
    db.sql(f"""
        UPDATE {CLASSIFICATION_TABLE}
        SET {CLEAN_COLUMN} = data_clean.{CLEAN_COLUMN}
        FROM data_clean
        WHERE {CLASSIFICATION_TABLE}.id = data_clean.id
    """)

    print(f"Done! '{CLEAN_COLUMN}' added to {CLASSIFICATION_TABLE}.")

    db.close()


if __name__ == '__main__':
    main()
