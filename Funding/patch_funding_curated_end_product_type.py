## One-off patch: pushes the 8 genuinely-corrected columns from the fixed last-report raw file
## (Funding2026_inscope2.xlsx) into funding_curated, matched by lrd_row_id's positional key.
## See C:\Users\tsdav\.claude\plans\steady-skipping-balloon.md for the full investigation/plan.
## Not part of the regular pipeline - safe to delete after use.

import duckdb
import pandas as pd

OLD_FILE = 'raw_data/Funding2026_inscope.xlsx'
NEW_FILE = 'raw_data/Funding2026_inscope2.xlsx'
DB_PATH = 'funding.db'
SHEET = 'Airtable copy'

# Columns confirmed to have genuine corrections (not formatting noise, not the wiped
# 2020-2033/Years active columns - those are excluded deliberately, see the plan).
CORRECTED_COLUMNS = [
    'End product type',
    'PI organisation region',
    'Sub-production pillar',
    'Production platform',
    'Currency',
    'Sub-funder',
    'End date',
    'duration (years)',
]

# 'Production platform' is renamed to 'AP pillar' at load time in S2's notebook (cell
# fd1d7802), so that's its actual name in funding_curated - read the source column under its
# original name, write to the target under this mapped name.
TARGET_COLUMN = {'Production platform': 'AP pillar'}

old = pd.read_excel(OLD_FILE, sheet_name=SHEET)
new = pd.read_excel(NEW_FILE, sheet_name=SHEET)
assert len(old) == len(new), f"row count mismatch: old={len(old)} new={len(new)}"

con = duckdb.connect(DB_PATH)

fc = con.sql("SELECT lrd_row_id FROM funding_curated").df()
is_plain = fc['lrd_row_id'].str.match(r'^\d+$')
valid_positions = set(fc.loc[is_plain, 'lrd_row_id'].astype(int))
print(f"{len(fc)} total rows in funding_curated, {len(valid_positions)} with a plain (LRD-origin) lrd_row_id")

for col in CORRECTED_COLUMNS:
    diff_mask = (old[col].astype(str) != new[col].astype(str)) & ~(old[col].isna() & new[col].isna())
    diff_positions = sorted(set(diff_mask[diff_mask].index) & valid_positions)

    if not diff_positions:
        print(f"{col}: no matching rows in funding_curated - skipped")
        continue

    patch_df = pd.DataFrame({
        'lrd_row_id': [str(p) for p in diff_positions],
        'new_value': [None if pd.isna(new.loc[p, col]) else new.loc[p, col] for p in diff_positions],
    })

    target_col = TARGET_COLUMN.get(col, col)
    before = con.sql(f'SELECT COUNT("{target_col}") AS n FROM funding_curated').df()['n'][0]
    con.register('patch_df', patch_df)
    con.sql(f'''
        UPDATE funding_curated
        SET "{target_col}" = patch_df.new_value
        FROM patch_df
        WHERE funding_curated.lrd_row_id = patch_df.lrd_row_id
    ''')
    con.unregister('patch_df')
    after = con.sql(f'SELECT COUNT("{target_col}") AS n FROM funding_curated').df()['n'][0]
    print(f'{col} -> {target_col}: updated {len(patch_df)} rows (non-null count {before} -> {after})')

con.close()
print("\nDone.")
