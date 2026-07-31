## One-off export: funding_curated (the combined last-report + grants-tracker dataset) plus
## country/year trend summaries, for a quick pre-meeting read on data coverage by country.
## Not part of the regular pipeline - safe to delete after use.

from datetime import datetime

import duckdb
import pandas as pd

DB_PATH = 'funding.db'
OUT_DIR = 'data_audit'

con = duckdb.connect(database=DB_PATH, read_only=True)
data = con.sql('SELECT * FROM funding_curated').df()
con.close()

print(f"{len(data)} rows loaded from 'funding_curated'")


def summarize(df, group_col):
    work = df.copy()
    work[group_col] = work[group_col].astype(object).where(work[group_col].notna(), 'Unknown')
    summary = work.groupby(group_col).agg(
        grant_count=('Title', 'size'),
        total_usd=('Total amount (USD)', 'sum'),
    ).reset_index()
    return summary.sort_values('grant_count', ascending=False).reset_index(drop=True)


# Same country under two spellings in the source data - merge for the country breakdown only
# (raw 'Combined Data' sheet keeps the original values for traceability).
_COUNTRY_ALIASES = {'United Kingdom': 'UK'}
country_data = data.copy()
for col in ('PI organisation country', 'Funder Country'):
    country_data[col] = country_data[col].replace(_COUNTRY_ALIASES)

by_pi_country = summarize(country_data, 'PI organisation country').rename(
    columns={'PI organisation country': 'PI organisation country', 'grant_count': 'grant_count_pi', 'total_usd': 'total_usd_pi'}
)
by_funder_country = summarize(country_data, 'Funder Country').rename(
    columns={'Funder Country': 'Funder Country', 'grant_count': 'grant_count_funder', 'total_usd': 'total_usd_funder'}
)
by_year = summarize(data, 'Year project started')
by_year['_sort'] = by_year['Year project started'].apply(lambda v: (1, 0) if v == 'Unknown' else (0, v))
by_year = by_year.sort_values('_sort').drop(columns='_sort').reset_index(drop=True)

timestamp = datetime.now().strftime('%y%m%d_%H%M')
out_path = f"{OUT_DIR}/{timestamp}_combined_report_for_meeting.xlsx"

with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
    data.to_excel(writer, sheet_name='Combined Data', index=False)

    # Country sheet: PI org country block, blank row, then Funder country block
    by_pi_country.to_excel(writer, sheet_name='By Country', index=False, startrow=0)
    gap = len(by_pi_country) + 2
    by_funder_country.to_excel(writer, sheet_name='By Country', index=False, startrow=gap)

    by_year.to_excel(writer, sheet_name='By Year', index=False)

print(f"\nSaved: {out_path}")

# --- Verification ---
check = pd.read_excel(out_path, sheet_name=None)
for sheet, df in check.items():
    print(f"Sheet '{sheet}': {len(df)} rows")

print("\nTop 5 PI organisation countries by grant count:")
print(by_pi_country.head(5).to_string(index=False))

print("\nGrants by year:")
print(by_year.to_string(index=False))

total_check = data['Total amount (USD)'].sum()
print(f"\nTotal USD across all {len(data)} rows: {total_check:,.0f}")
