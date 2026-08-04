## Shared helper functions for Funding_analysis.ipynb.

import json
import time
import urllib.error
import urllib.request

import pandas as pd

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/json, text/plain, */*',
}

YEAR_COLS = list(range(2020, 2036))


def split_multi_value(df, col, delimiter=';'):
    """Explode a multi-value column (e.g. 'Meat; Dairy') into one row per DISTINCT value found in
    each cell, adding a '_weight' column = 1 / n_distinct so a per-row amount can be split evenly
    across the values assigned to it (duplicate values within the same cell, e.g. 'Germany;
    Germany', collapse to one distinct entry and don't inflate the split). Rows with a blank value
    in `col` are dropped - nothing to attribute them to. The original `col` is left untouched; the
    new column is called f'{col}_split'."""
    work = df[df[col].notna()].copy()
    distinct_values = work[col].astype(str).apply(
        lambda s: sorted(set(v.strip() for v in s.split(delimiter) if v.strip()))
    )
    keep = distinct_values.apply(len) > 0
    work = work[keep].copy()
    distinct_values = distinct_values[keep]
    work[f'{col}_split'] = distinct_values
    work['_weight'] = distinct_values.apply(lambda v: 1.0 / len(v))
    return work.explode(f'{col}_split')


def weighted_sum(df, amount_col, weight_col='_weight'):
    """Sum amount_col x weight_col - the pattern every split_multi_value-based aggregation uses."""
    return (df[amount_col] * df[weight_col]).sum()


def _fetch_json(url, timeout=15):
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def fetch_worldbank_indicator(indicator_code, year):
    """Fetch a World Bank indicator for every country for a given year. Returns {country_name:
    value}, using World Bank's own country names (aliased to match funding_curated's naming by
    the caller where they differ, e.g. 'Slovakia' -> 'Slovak Republic')."""
    url = (
        f'https://api.worldbank.org/v2/country/all/indicator/{indicator_code}'
        f'?format=json&date={year}&per_page=400'
    )
    data = _fetch_json(url)
    if len(data) < 2 or data[1] is None:
        return {}
    return {
        entry['country']['value']: entry['value']
        for entry in data[1]
        if entry['value'] is not None
    }


# funding_curated's country naming vs World Bank's / OECD's - only needed where they genuinely
# differ (verified against a live pull of each source, not guessed).
COUNTRY_NAME_TO_WB = {
    'Slovakia': 'Slovak Republic',
    'South Korea': 'Korea, Rep.',
    'Russia': 'Russian Federation',
    'Turkey': 'Turkiye',
}
COUNTRY_NAME_TO_OECD = {
    'Slovakia': 'Slovak Republic',
    'South Korea': 'Korea',
    'Russia': 'Russia',  # OECD's own label is also just 'Russia'
    'Turkey': 'Türkiye',
}


def lookup_country_value(country, value_dict, alias_map=COUNTRY_NAME_TO_WB):
    """value_dict is keyed by the source's own naming (World Bank or OECD) - resolve
    funding_curated's country name to it via alias_map where they differ, otherwise try the name
    directly. Returns None if the country isn't in that source at all (e.g. 'EU', 'Global')."""
    resolved_name = alias_map.get(country, country)
    return value_dict.get(resolved_name)


def fetch_oecd_gerd(year, lookback_years=3):
    """OECD Gross Domestic Expenditure on R&D (GERD), current-price USD PPP, most recent available
    year at or before `year` per country (some countries lag a year or two on reporting - looking
    back a few years and taking the latest available avoids leaving them blank unnecessarily).
    Returns {country_name: value_usd}, using OECD's own 'Reference area' naming (matches
    funding_curated directly for most European countries; resolve the rest via
    COUNTRY_NAME_TO_OECD through lookup_country_value)."""
    import io

    url = (
        'https://sdmx.oecd.org/public/rest/data/OECD.STI.STP,DSD_MSTI@DF_MSTI,/all'
        f'?startPeriod={year - lookback_years}&endPeriod={year}&format=csvfilewithlabels'
    )
    req = urllib.request.Request(url, headers=REQUEST_HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode('utf-8')
    raw = pd.read_csv(io.StringIO(text))

    gerd = raw[
        (raw['MEASURE'] == 'G') & (raw['UNIT_MEASURE'] == 'USD_PPP') &
        (raw['PRICE_BASE'] == 'V') & (raw['TRANSFORMATION'] == '_Z')
    ].copy()
    gerd['value'] = gerd['OBS_VALUE'] * (10 ** gerd['UNIT_MULT'])
    latest = gerd.sort_values('TIME_PERIOD').groupby('Reference area').tail(1)
    return dict(zip(latest['Reference area'], latest['value']))


def fetch_usd_to_eur_rate(date_str):
    """Historical USD->EUR rate on date_str (YYYY-MM-DD) via Frankfurter - needed to convert World
    Bank's (USD) and OECD's (USD PPP) figures into EUR for the per-capita/GDP/R&D-spend columns.
    Frankfurter's bot protection rejects urllib's default User-Agent (403) regardless of network -
    REQUEST_HEADERS above works around that (same fix S7_data_transformation.ipynb uses)."""
    url = f'https://api.frankfurter.app/{date_str}?from=USD&to=EUR'
    data = _fetch_json(url, timeout=10)
    return data['rates']['EUR']


def yoy_growth(series):
    """Year-over-year % change: (current - previous) / previous, matching the colleagues' template
    formula exactly. Returns a Series aligned with `series`, first entry NaN (no prior year)."""
    return series.pct_change()
