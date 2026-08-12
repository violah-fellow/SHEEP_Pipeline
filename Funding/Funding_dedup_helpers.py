## Shared helper functions for S2_grant_deduplication.ipynb.
## Factors out logic that was duplicated ~4-5x inline in the original grant_deduplication.ipynb
## prototype: gap-filling, highlighted-diff audit exports, and the export-for-review /
## apply-reviewed-decisions round trip used at each of the 4 manual title-match review points.

import re
from pathlib import Path

import pandas as pd
import pycountry

def is_empty(val):
    if pd.isna(val):  # check for NaN / None / pd.NA
        return True
    return str(val).strip() == ''  # also treat blank strings as empty


# Abbreviations/typos/alternate names seen across raw sources that should read as one canonical
# name everywhere - keyed lowercase for case-insensitive lookup.
COUNTRY_ALIASES = {
    'usa': 'United States', 'u.s.a.': 'United States', 'u.s.': 'United States',
    'uk': 'United Kingdom',
    'czech republic': 'Czechia',  # both seen in the raw data - Czechia is the more common form here
    'finalnd': 'Finland',  # typo seen in raw Dimensions PI-organisation data
}

# Region for every country name this pipeline is likely to see, grouped into the 8 categories
# GFI's reporting uses (Africa / Asia-Asia Pacific / Europe / North America / South America /
# Oceania / Middle East / Global). Built from standard geography, cross-checked against every
# already-correctly-labeled row in funding_curated at the time this was written, matching this
# dataset's own established conventions where they diverge from strict geography: Russia and
# Turkey -> Europe (every existing row already does this - both have deep Council of
# Europe/OECD-Europe ties); Australia/New Zealand/Pacific nations -> 'Asia/Asia Pacific' rather
# than 'Oceania' (every existing Australia row already does this; 'Oceania' was only used
# inconsistently twice). 'EU' and 'Global' are the two pseudo-country values already used in
# 'Funder Country' - not real countries, but need a region each all the same.
_EUROPE = [
    'Austria', 'Belgium', 'Bulgaria', 'Croatia', 'Cyprus', 'Czechia', 'Denmark', 'Estonia',
    'Finland', 'France', 'Germany', 'Greece', 'Hungary', 'Iceland', 'Ireland', 'Italy', 'Latvia',
    'Liechtenstein', 'Lithuania', 'Luxembourg', 'Malta', 'Moldova', 'Monaco', 'Montenegro',
    'Netherlands', 'North Macedonia', 'Norway', 'Poland', 'Portugal', 'Romania', 'Russia',
    'San Marino', 'Serbia', 'Slovakia', 'Slovenia', 'Spain', 'Sweden', 'Switzerland', 'Turkey',
    'Ukraine', 'United Kingdom', 'Bosnia and Herzegovina', 'Albania', 'Belarus',
]
_NORTH_AMERICA = [
    'United States', 'Canada', 'Mexico', 'Bermuda', 'Greenland',
]
_SOUTH_AMERICA = [
    'Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru', 'Uruguay', 'Paraguay', 'Bolivia',
    'Ecuador', 'Venezuela', 'Guyana', 'Suriname', 'Costa Rica', 'Panama', 'Guatemala',
    'Honduras', 'El Salvador', 'Nicaragua', 'Cuba', 'Dominican Republic', 'Jamaica',
]
_ASIA_PACIFIC = [
    'China', 'Japan', 'South Korea', 'North Korea', 'India', 'Indonesia', 'Malaysia', 'Thailand',
    'Vietnam', 'Nepal', 'Philippines', 'Singapore', 'Taiwan', 'Bangladesh', 'Sri Lanka',
    'Pakistan', 'Afghanistan', 'Myanmar', 'Cambodia', 'Laos', 'Mongolia', 'Kazakhstan',
    'Uzbekistan', 'Australia', 'New Zealand', 'Fiji', 'Papua New Guinea', 'Bhutan', 'Maldives',
    'Brunei', 'Timor-Leste',
]
_MIDDLE_EAST = [
    'Israel', 'Saudi Arabia', 'United Arab Emirates', 'Qatar', 'Kuwait', 'Bahrain', 'Oman',
    'Jordan', 'Lebanon', 'Iraq', 'Iran', 'Syria', 'Yemen',
]
_AFRICA = [
    'South Africa', 'Egypt', 'Nigeria', 'Kenya', 'Ghana', 'Morocco', 'Algeria', 'Tunisia',
    'Libya', 'Ethiopia', 'Tanzania', 'Uganda', 'Rwanda', 'Senegal', 'Ivory Coast', 'Cameroon',
    'Zimbabwe', 'Zambia', 'Botswana', 'Namibia', 'Mozambique', 'Djibouti',
]

COUNTRY_TO_REGION = {
    **{c: 'Europe' for c in _EUROPE},
    **{c: 'North America' for c in _NORTH_AMERICA},
    **{c: 'South America' for c in _SOUTH_AMERICA},
    **{c: 'Asia/Asia Pacific' for c in _ASIA_PACIFIC},
    **{c: 'Middle East' for c in _MIDDLE_EAST},
    **{c: 'Africa' for c in _AFRICA},
    'EU': 'Europe',
    'Global': 'Global',
}


def derive_region(country_val, delimiter):
    """Derive a region value from a (possibly multi-value) country value - splits on `delimiter`,
    maps each distinct token through COUNTRY_TO_REGION, deduplicates, and rejoins with ', '
    (matching the multi-region format already used in funding_curated, e.g. 'Europe, North
    America'). Returns None if country_val is blank or none of its tokens have a known region -
    a real gap, not an error, since a handful of very small/rare countries aren't in the table."""
    if is_empty(country_val):
        return None
    tokens = [t.strip() for t in str(country_val).split(delimiter) if t.strip()]
    regions = []
    for t in tokens:
        region = COUNTRY_TO_REGION.get(t)
        if region and region not in regions:
            regions.append(region)
    return ', '.join(regions) if regions else None


# Every name/common-name pycountry recognizes, plus a few seen in the raw data that pycountry
# doesn't carry under these exact names - used to distinguish a real country from junk (person
# names, placeholders like 'TBD'/'Many'/'10 countries') in clean_pi_country_cell below.
VALID_COUNTRY_NAMES = set()
for _c in pycountry.countries:
    VALID_COUNTRY_NAMES.add(_c.name.strip().lower())
    if hasattr(_c, 'common_name'):
        VALID_COUNTRY_NAMES.add(_c.common_name.strip().lower())
VALID_COUNTRY_NAMES.update({'czech republic', 'russia', 'turkey', 'uk'})


def clean_pi_country_cell(val):
    """Clean a (possibly multi-value) PI-organisation-country cell: splits on both ';' and ','
    (a handful of raw values pack a second country into what should be its own entry, e.g.
    'Denmark, Sweden' appearing as a single semicolon-separated token - splitting on both
    delimiters uniformly handles this with no special-casing needed, since normal entries never
    contain a comma anyway), normalizes aliases/typos via COUNTRY_ALIASES, drops any token that
    isn't a real country (catches junk seen in the raw data - person names, 'TBD', 'Many',
    '10 countries'), deduplicates the remaining distinct tokens, and rejoins with '; '. Returns
    None if nothing valid remains."""
    if is_empty(val):
        return val
    seen = []
    for tok in re.split(r'[;,]', str(val)):
        tok = tok.strip()
        if not tok:
            continue
        normalized = COUNTRY_ALIASES.get(tok.lower(), tok)
        if normalized.strip().lower() not in VALID_COUNTRY_NAMES:
            continue
        if normalized not in seen:
            seen.append(normalized)
    return '; '.join(seen) if seen else None


def normalize_country_name(val):
    """Normalize every semicolon/comma-delimited token in val through COUNTRY_ALIASES (not just
    the first) - needed for multi-country fields like 'Funder Country' ('Denmark, UK' etc.).
    Tokens with no matching alias are returned unchanged, including their original spacing -
    tokens that DO match keep their original leading/trailing whitespace too, so 'Denmark, UK'
    becomes 'Denmark, United Kingdom', not 'Denmark,United Kingdom'."""
    if is_empty(val):
        return val
    parts = re.split(r'([;,])', str(val))
    for i in range(0, len(parts), 2):
        token = parts[i]
        alias = COUNTRY_ALIASES.get(token.strip().lower())
        if alias:
            leading = token[:len(token) - len(token.lstrip())]
            trailing = token[len(token.rstrip()):]
            parts[i] = leading + alias + trailing
    return ''.join(parts)


def is_zero(val):
    """Return True if val is numerically zero - used to overwrite placeholder 0s in funding columns."""
    if is_empty(val):
        return False
    try:
        return float(val) == 0
    except (ValueError, TypeError):
        return False


def is_real_nonzero(val):
    """Return True if val is a genuine, populated, nonzero number - the opposite of is_empty(val)
    or is_zero(val) combined. Used by completeness checks that need to tell "a real amount is
    here" apart from both blank and placeholder-0, without the caller having to combine the two
    checks (and get the negation backwards) every time."""
    if is_empty(val):
        return False
    try:
        return float(val) != 0
    except (ValueError, TypeError):
        return False


def normalize_title(val):
    if is_empty(val):
        return None
    return str(val).strip().lower()


# funding_curated's target format for 'AP pillar' is the full pillar name (Plant-based/
# Fermentation/Cultivated/Cross-cutting) - confirmed by S6_LLM_labelling.py's CODE_TO_AP_PILLAR,
# which always produces these 4 exact strings when promoting a new Dimensions grant. The short
# PB/F/CM/CC codes are internal-only (funding_classified's pillar_curated / the LLM tool schema),
# never this column.
CANONICAL_AP_PILLARS = {
    'plant-based': 'Plant-based', 'fermentation': 'Fermentation',
    'cultivated': 'Cultivated', 'cross-cutting': 'Cross-cutting',
}


def normalize_ap_pillar(val):
    """Returns (normalized_value, resolved: bool). Unresolved rows keep their original value so
    a human reviewing the export sees what actually needs deciding, not an empty cell.
    Handles: casing variants of the 4 canonical pillars; a trailing end-product descriptor
    leaked into the pillar field (e.g. 'Plant-based meat' -> 'Plant-based'); comma-joined
    multi-pillar combos where every token is a canonical pillar (-> 'Cross-cutting', matching the
    convention already used in S3_LLM_scope.py's derive_pillar for >1 true pillar flag).
    Anything else (blank, 'Agnostic', an unrecognized token) is left unresolved."""
    if is_empty(val):
        return val, False

    text = str(val).strip()
    for suffix in (' meat', ' seafood'):
        if text.lower().endswith(suffix):
            text = text[: -len(suffix)]
            break

    tokens = [t.strip().lower() for t in text.split(',') if t.strip()]
    if not tokens or not all(t in CANONICAL_AP_PILLARS for t in tokens):
        return val, False

    distinct = {CANONICAL_AP_PILLARS[t] for t in tokens}
    if len(distinct) == 1:
        return next(iter(distinct)), True
    return 'Cross-cutting', True


def extract_year(val):
    """Extract just the calendar year from a messy end-date value - handles the Grants Tracker
    raw export's mix of datetime objects, 'dd/mm/yyyy' strings, and ISO strings (verified: this
    parses 1251/1251 of the real 'INT_End Date' column with dayfirst=True). Returns None on
    NaT/blank rather than raising, so callers can apply it across a whole column safely."""
    if is_empty(val):
        return None
    parsed = pd.to_datetime(val, dayfirst=True, errors='coerce')
    if pd.isna(parsed):
        return None
    return parsed.year


def derive_duration_and_years_active(start_year, end_year):
    """(duration:int|None, years_active:str|None) from a grant's start/end year, using the
    'inclusive calendar years spanned' convention already established in Funding2026_inscope.xlsx
    (e.g. Aug 2022 - Aug 2024 counts as 3 years: 2022, 2023, 2024 - verified against 1507/1511
    existing rows). Returns (None, None) if either year is missing or end_year < start_year (an
    internal-consistency guard, not a plausibility bound - out-of-range years are otherwise
    accepted as-is, by design)."""
    if is_empty(start_year) or is_empty(end_year):
        return None, None
    start_year, end_year = int(start_year), int(end_year)
    if end_year < start_year:
        return None, None
    years = list(range(start_year, end_year + 1))
    return len(years), ', '.join(str(y) for y in years)


def assign_stable_row_id(df, id_col):
    """Stamp a surrogate row ID from the row's position in the just-loaded raw file, as an
    explicit column (not the pandas index) - so it survives later reset_index/filtering.
    Call immediately after loading a source, before any filtering, so the ID reflects the
    row's position in the untouched raw pull."""
    df = df.copy()
    df[id_col] = df.index
    return df


def gap_fill(source_row, target_df, target_idx, col_map, funding_cols, changed_indices):
    """Fill empty cells in target_df.loc[target_idx] from source_row, following col_map
    (source_col -> target_col). Never overwrites a non-empty value, except columns listed
    in funding_cols, where an existing 0 is treated as fillable. Returns the count of cells
    filled and records target_idx in changed_indices if anything was filled."""
    cells_filled = 0
    for src_col, tgt_col in col_map.items():
        if src_col not in source_row.index or tgt_col not in target_df.columns:
            continue
        src_val = source_row[src_col]
        if is_empty(src_val):
            continue
        target_val = target_df.at[target_idx, tgt_col]
        overwrite_zero = tgt_col in funding_cols and is_zero(target_val)
        if is_empty(target_val) or overwrite_zero:
            target_df.at[target_idx, tgt_col] = src_val
            changed_indices.add(target_idx)
            cells_filled += 1
    return cells_filled


def _split_first_rest_fill(value, target_df, target_idx, first_col, rest_col, changed_indices):
    """Split a semicolon-separated value; fill the first entry into first_col (e.g. PI /
    lead researcher) and the remainder into rest_col (e.g. collaborators), only where those
    target cells are currently empty."""
    if is_empty(value):
        return 0
    parts = [p.strip() for p in str(value).split(';') if p.strip()]
    if not parts:
        return 0

    filled = 0
    first_empty = is_empty(target_df.at[target_idx, first_col])
    rest_empty = is_empty(target_df.at[target_idx, rest_col])

    if first_empty:
        target_df.at[target_idx, first_col] = parts[0]
        changed_indices.add(target_idx)
        filled += 1
        if rest_empty and len(parts) > 1:
            target_df.at[target_idx, rest_col] = '; '.join(parts[1:])
            filled += 1
    elif rest_empty and len(parts) > 1:
        target_df.at[target_idx, rest_col] = '; '.join(parts[1:])
        changed_indices.add(target_idx)
        filled += 1
    return filled


def gap_fill_researchers_and_orgs(
    source_row, target_df, target_idx,
    pi_col, collab_col, org_pi_col, org_collab_col, changed_indices,
    researchers_field='Researchers', org_field='Research Organization - standardized',
):
    """Split Dimensions' semicolon-joined Researchers / Research Organization - standardized
    fields: first entry -> PI / lead org, remainder -> collaborators. Shared by every source
    that gap-fills from Dimensions data (last-year data, grants tracker)."""
    filled = 0
    filled += _split_first_rest_fill(
        source_row.get(researchers_field), target_df, target_idx, pi_col, collab_col, changed_indices
    )
    filled += _split_first_rest_fill(
        source_row.get(org_field), target_df, target_idx, org_pi_col, org_collab_col, changed_indices
    )
    return filled


# Funding columns where gap_fill() treats an existing 0 as fillable (see its own funding_cols
# param) - export_highlighted_diff needs the same list so it doesn't miss a 0-to-real-value change
# that gap_fill() legitimately made. Covers every Total/Gov & NP variant (native/USD/EUR) used
# across the pipeline's various merge paths, since each call site gap-fills a different subset.
FUNDING_HIGHLIGHT_COLS = {
    'Total amount', 'Total amount (USD)', 'Total amount (EUR)',
    'Gov contribution', 'Gov contribution (USD)',
    'Gov & NP contribution', 'Gov & NP contribution (USD)', 'Gov & NP contribution (EUR)',
    'INT_Total amount (actual currency)', 'EXT_Total amount (USD)',
    'INT_Gov contribution (actual currency)', 'EXT_Gov contribution (USD)',
}


def export_highlighted_diff(before_df, after_df, changed_indices, out_path):
    """Save an Excel audit trail: rows in changed_indices, cells that went from empty (in
    before_df) to filled (in after_df) highlighted green. Returns the (unstyled) view.
    A funding column (FUNDING_HIGHLIGHT_COLS) going from a literal 0 to a real value also counts
    as a highlighted change, matching gap_fill()'s own zero-is-fillable rule for those columns -
    otherwise a real funding backfill (0 -> real amount) silently isn't highlighted, since 0 isn't
    "empty" under the generic is_empty() check."""
    view = after_df.loc[sorted(changed_indices)]

    def _highlight(data):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for idx in data.index:
            for col in data.columns:
                before_val = before_df.at[idx, col]
                after_val = data.at[idx, col]
                became_filled = is_empty(before_val) and not is_empty(after_val)
                became_filled_from_zero = (
                    col in FUNDING_HIGHLIGHT_COLS and is_zero(before_val)
                    and not is_empty(after_val) and not is_zero(after_val)
                )
                if became_filled or became_filled_from_zero:
                    styles.at[idx, col] = 'background-color: #c6efce; color: #276221'
        return styles

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    styled = view.style.apply(_highlight, axis=None)
    styled.to_excel(out_path, index=True)
    return view


def _build_match_key(df, id_cols):
    """Concatenate id_cols into a single '__'-joined string key, column-by-column rather than
    via a row-wise .astype(str).agg(join, axis=1) - that pattern silently leaves NaN as an
    actual float (not the string 'nan') in newer pandas versions when the row also contains
    string columns, which then crashes str.join. fillna('NA') first sidesteps that entirely."""
    key = df[id_cols[0]].fillna('NA').astype(str)
    for col in id_cols[1:]:
        key = key + '__' + df[col].fillna('NA').astype(str)
    return key


def export_for_review(matches_df, id_cols, out_path, decision_col='is_true_match', default=True):
    """Export candidate matches for human review. Builds a composite match_key from id_cols
    (so the decision can be re-applied by key rather than by fragile row position), pre-fills
    decision_col with `default`, and writes a CSV. Returns the exported dataframe.
    No-ops (no file written) when matches_df is empty - nothing to review."""
    if len(matches_df) == 0:
        print("No candidate matches to review - skipping export.")
        return matches_df.copy()

    df = matches_df.copy()
    df['match_key'] = _build_match_key(df, id_cols)
    df[decision_col] = default
    df = df[['match_key'] + [c for c in df.columns if c != 'match_key']]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} candidate matches for review -> {out_path}")
    return df


def _coerce_bool(val, default_on_missing=True):
    if isinstance(val, bool):
        return val
    if pd.isna(val):
        return default_on_missing
    return str(val).strip().lower() in ('true', '1', 'yes', 'y', 't')


def apply_reviewed_decisions(matches_df, reviewed_csv_path, id_cols, decision_col='is_true_match'):
    """Read a human-edited copy of an export_for_review CSV back in, rejoin by match_key
    (not row position), and split matches_df into (confirmed, rejected) based on decision_col.
    Raises if any match_key present in matches_df is missing from the reviewed file, to guard
    against a stale or mismatched reviewed file being applied against a different run.
    No-ops (no file read) when matches_df is empty - export_for_review skipped writing one."""
    if len(matches_df) == 0:
        print("No candidate matches were exported for review - skipping reviewed-file read.")
        empty = matches_df.copy()
        return empty, empty

    reviewed = pd.read_csv(reviewed_csv_path)
    if 'match_key' not in reviewed.columns:
        raise ValueError(
            f"'{reviewed_csv_path}' has no 'match_key' column - was it exported by export_for_review?"
        )
    # match_key is always built as a string (see _build_match_key), but a purely-numeric id_cols
    # (e.g. a single int column) round-trips through CSV as int64 by pandas' default type
    # inference - cast back to string so the comparison/join below isn't silently comparing
    # strings against ints (which would report every key as missing).
    reviewed['match_key'] = reviewed['match_key'].astype(str)

    working = matches_df.copy()
    working['match_key'] = _build_match_key(working, id_cols)

    missing = set(working['match_key']) - set(reviewed['match_key'])
    if missing:
        raise ValueError(
            f"{len(missing)} match_key(s) in matches_df are missing from '{reviewed_csv_path}' - "
            f"the reviewed file may be stale or from a different run. "
            f"Missing keys (first 5): {sorted(missing)[:5]}"
        )

    decisions = reviewed.set_index('match_key')[decision_col].apply(_coerce_bool)
    working[decision_col] = working['match_key'].map(decisions)

    confirmed = working[working[decision_col] == True].drop(columns=['match_key', decision_col])
    rejected = working[working[decision_col] != True].drop(columns=['match_key', decision_col])
    return confirmed.reset_index(drop=True), rejected.reset_index(drop=True)


def apply_reviewed_values(matches_df, reviewed_csv_path, id_cols, value_col):
    """Read a human-edited copy of an export_for_review CSV back in, rejoin by match_key (not
    row position), and return matches_df with value_col overwritten by the reviewed file's
    values. Unlike apply_reviewed_decisions, this is for a free-text corrected-value column, not
    a boolean confirm/reject - no coercion, no confirmed/rejected split. Raises if any match_key
    present in matches_df is missing from the reviewed file, to guard against a stale or
    mismatched reviewed file being applied against a different run. No-ops (no file read) when
    matches_df is empty - export_for_review skipped writing one."""
    if len(matches_df) == 0:
        print("No candidate rows were exported for review - skipping reviewed-file read.")
        return matches_df.copy()

    reviewed = pd.read_csv(reviewed_csv_path)
    if 'match_key' not in reviewed.columns:
        raise ValueError(
            f"'{reviewed_csv_path}' has no 'match_key' column - was it exported by export_for_review?"
        )
    # match_key is always built as a string (see _build_match_key), but a purely-numeric id_cols
    # (e.g. a single int column) round-trips through CSV as int64 by pandas' default type
    # inference - cast back to string so the comparison/join below isn't silently comparing
    # strings against ints (which would report every key as missing).
    reviewed['match_key'] = reviewed['match_key'].astype(str)

    working = matches_df.copy()
    working['match_key'] = _build_match_key(working, id_cols)

    missing = set(working['match_key']) - set(reviewed['match_key'])
    if missing:
        raise ValueError(
            f"{len(missing)} match_key(s) in matches_df are missing from '{reviewed_csv_path}' - "
            f"the reviewed file may be stale or from a different run. "
            f"Missing keys (first 5): {sorted(missing)[:5]}"
        )

    values = reviewed.set_index('match_key')[value_col]
    working[value_col] = working['match_key'].map(values)
    return working.drop(columns=['match_key']).reset_index(drop=True)


def invalid_category_tokens(value, valid_categories, delimiter):
    """Split value on delimiter and return whichever stripped tokens aren't in valid_categories -
    empty list means every token matched. Generic on purpose (doesn't know about rescat/end_product/
    award_purpose specifically) so callers pass in whatever category list/delimiter applies for
    that stage - used to catch typos in freehand manual-review entries (e.g. 'Impact assessment'
    vs the real 'Impact Assessments') before they get written into the data."""
    if is_empty(value):
        return []
    tokens = [t.strip() for t in str(value).split(delimiter) if t.strip()]
    valid_set = set(valid_categories)
    return [t for t in tokens if t not in valid_set]
