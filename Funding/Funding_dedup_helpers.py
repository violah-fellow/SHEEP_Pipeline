## Shared helper functions for S2_grant_deduplication.ipynb.
## Factors out logic that was duplicated ~4-5x inline in the original grant_deduplication.ipynb
## prototype: gap-filling, highlighted-diff audit exports, and the export-for-review /
## apply-reviewed-decisions round trip used at each of the 4 manual title-match review points.

from pathlib import Path

import pandas as pd


def is_empty(val):
    if pd.isna(val):  # check for NaN / None / pd.NA
        return True
    return str(val).strip() == ''  # also treat blank strings as empty


def is_zero(val):
    """Return True if val is numerically zero - used to overwrite placeholder 0s in funding columns."""
    if is_empty(val):
        return False
    try:
        return float(val) == 0
    except (ValueError, TypeError):
        return False


def normalize_title(val):
    if is_empty(val):
        return None
    return str(val).strip().lower()


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


def export_highlighted_diff(before_df, after_df, changed_indices, out_path):
    """Save an Excel audit trail: rows in changed_indices, cells that went from empty (in
    before_df) to filled (in after_df) highlighted green. Returns the (unstyled) view."""
    view = after_df.loc[sorted(changed_indices)]

    def _highlight(data):
        styles = pd.DataFrame('', index=data.index, columns=data.columns)
        for idx in data.index:
            for col in data.columns:
                if is_empty(before_df.at[idx, col]) and not is_empty(data.at[idx, col]):
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
