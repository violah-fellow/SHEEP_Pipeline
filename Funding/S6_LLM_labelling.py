## LLM labelling script: assign Research category, End product, Award purpose, and Subpillar
## labels to in-scope grants via Claude's Batch API, then promote successfully-labelled
## Dimensions-sourced grants from funding_classified into funding_curated, and fill in missing
## labels for grants already sitting in funding_curated (Airtable's Grants Tracker, last year's
## report data, and any other historical source never touched by the Dimensions pipeline).
## Each enabled stage submits its own batch (combining both its funding_classified and
## funding_curated pools); all enabled stages' batches are submitted together and polled in one
## shared loop, rather than run one at a time to completion.
## Wired into pipeline_funding.py as Step 5 ("llm_labelling").

import os
import re

# CONFIG
# edit parameters for this run here

# Anthropic API
# path to API key
KEY_PATH = '../.env'

# Database
# path to DuckDB database (self-contained in Pipeline/Funding, mirrors Publications)
DB_PATH = 'funding.db'
# Dimensions-sourced staging table (populated by S1/S3, curated by S5)
CLASSIFICATION_TABLE = 'funding_classified'
# final GFI-tracker-schema dataset (built/grown by S2)
CURATED_TABLE = 'funding_curated'
SCOPE_COL = 'scope_curated'
PILLAR_COL = 'pillar_curated'

# Which stages to run this session - disable any to skip it entirely (submission, polling, writing)
RUN_RESCAT       = True
RUN_ENDPRODUCT   = True
RUN_AWARDPURPOSE = True
RUN_SUBPILLAR    = True

# 'new_only': only label rows never yet attempted for a given stage (funding_classified rows with
# that stage's own status column null, funding_curated rows with that stage's own target column
# blank) - checked independently per stage, so a grants-tracker row missing only one label is only
# sent through that one stage's batch.
# 'all': ignore prior status/values and relabel every eligible row for every enabled stage,
# overwriting whatever's there. Still excludes grants already promoted into funding_curated from
# funding_classified's pool (relabelling their staging-table copy would never reach the curated row
# anyway, since only promotion of a not-yet-promoted grant carries a label across).
LABEL_SCOPE = 'new_only'

# LLM
# one system prompt per pillar for rescat and subpillar - each asks for an independent TRUE/FALSE
# per category (subpillar's Cross-cutting pillar instead asks for a single choice - see
# SUBPILLAR_SCHEMA_MODE). End product and Award purpose use one shared prompt across all pillars.
PROMPT_PATHS = {
    'PB': 'llm_prompts/rescat_prompt_grants_PB.md',
    'F':  'llm_prompts/rescat_prompt_grants_F.md',
    'CM': 'llm_prompts/rescat_prompt_grants_CM.md',
    'CC': 'llm_prompts/rescat_prompt_grants_CC.md',
}
PROMPT_PATH_ENDPRODUCT   = 'llm_prompts/endproduct_prompt_grants.md'
PROMPT_PATH_AWARDPURPOSE = 'llm_prompts/awardpurpose_prompt_grants.md'
SUBPILLAR_PROMPT_PATHS = {
    'F':  'llm_prompts/subpillar_prompt_grants_F.md',
    'PB': 'llm_prompts/subpillar_prompt_grants_PB.md',
    'CC': 'llm_prompts/subpillar_prompt_grants_CC.md',
}

LLM_MODEL_LABEL = 'claude-sonnet-4-6'  # or 'claude-haiku-4-5' for cheap test runs
MAX_TOKENS = 300
TEMPERATURE = 0.0

# Batch
# directory for batch submission metadata
BATCH_DIR = 'batch_jobs'
# how often to check whether the batches have finished
POLL_INTERVAL_SECONDS = 600
# None = auto-resume the newest incomplete labelling run, or mint a fresh timestamped one
RUN_LABEL = None

# Output
# directory for the Excel copy of the final, updated funding_curated dataset
OUTPUT_DIR = 'data_output'

# START OF SCRIPT

# Category lists per pillar - verbatim from llm_prompts/rescat_prompt_grants_{PILLAR}.md
PB_CATS = ["Crop development", "Strain development", "Ingredient optimisation", "End product formulation",
           "Texturization methods", "Food safety & quality", "Health & nutrition",
           "Consumer & market research", "Impact Assessments", "Other"]
F_CATS = ["Feedstocks", "Target molecule selection", "Strain development", "Bioprocess design",
          "Ingredient optimisation", "End product formulation", "Texturization methods",
          "Food safety & quality", "Health & nutrition", "Consumer & market research",
          "Impact Assessments", "Other"]
CM_CATS = ["Cell line development", "Cell culture media", "Bioprocess design", "Scaffolding",
           "End product formulation", "Food safety & quality", "Health & nutrition",
           "Consumer & market research", "Impact Assessments", "Other"]
CC_CATS = ["Crop development", "Cell line development", "Strain development", "Target molecule selection",
           "Cell culture media", "Feedstocks", "Bioprocess design", "Scaffolding", "Ingredient optimisation",
           "End product formulation", "Texturization methods", "Food safety & quality", "Health & nutrition",
           "Consumer & market research", "Impact Assessments", "Other"]
PILLAR_CATS = {'PB': PB_CATS, 'F': F_CATS, 'CM': CM_CATS, 'CC': CC_CATS}

# End product categories - verbatim from llm_prompts/endproduct_prompt_grants.md
ENDPRODUCT_CATS = [
    "Meat", "Fish and seafood", "Milk and milk proteins", "Yoghurt and fermented dairy",
    "Cheese", "Cream and ice cream", "Infant formula", "Dairy", "Agnostic",
    "Chocolate, desserts, and confectionery", "Eggs and egg proteins",
    "Spreads, sauces, and condiments",
]
DAIRY_SUBCATS = ["Milk and milk proteins", "Yoghurt and fermented dairy", "Cheese",
                  "Cream and ice cream", "Infant formula"]

# Award purpose categories - verbatim from llm_prompts/awardpurpose_prompt_grants.md
AWARDPURPOSE_CATS = [
    "Research and development", "Education and training", "Networking",
    "Equipment and infrastructure", "Research infrastructure",
]

# Subpillar categories per pillar - verbatim from llm_prompts/subpillar_prompt_grants_{PILLAR}.md.
# Cultivated has no subpillar and is never included here.
SUBPILLAR_CATS = {
    'F':  ["BF", "PF"],
    'PB': ["Traditional fermentation"],
    'CC': ["Broad R&D", "Technical research", "Socioeconomic"],
}
# F/PB ask for an independent TRUE/FALSE per category; CC asks for a single choice (a grant can
# only be one type of research, not several at once).
SUBPILLAR_SCHEMA_MODE = {'F': 'multi_label', 'PB': 'multi_label', 'CC': 'single_label'}

# Maps funding_curated's 'AP pillar' values (full names, Grants Tracker origin) to the same
# short pillar codes funding_classified's pillar_curated already uses. Lowercased for lookup
# so case variants (e.g. a stray "Plant-Based") still resolve.
AP_PILLAR_TO_CODE = {
    'plant-based': 'PB',
    'fermentation': 'F',
    'cultivated': 'CM',
    'cross-cutting': 'CC',
}
# Reverse of the above - used at promotion time so a newly-promoted Dimensions grant gets 'AP
# pillar' set too. Without this, a promoted row's pillar is unrecoverable from funding_curated
# alone, so no later run's Pool B could ever pillar-match it for rescat or Subpillar backfilling.
CODE_TO_AP_PILLAR = {'PB': 'Plant-based', 'F': 'Fermentation', 'CM': 'Cultivated', 'CC': 'Cross-cutting'}

# Column mapping from funding_classified's Dimensions-derived schema into funding_curated's
# tracker schema - copied verbatim from S2_grant_deduplication.ipynb's col_map, so promoted
# rows land in the same columns S2 itself would gap-fill into.
PROMOTE_COL_MAP = {
    'Title translated':                              'Title',
    'Title':                                         'Original title',
    'Abstract translated':                           'Abstract',
    'Total amount':                                  'Total amount',
    'Currency':                                      'Currency',
    'Total amount (USD)':                            'Total amount (USD)',
    'Total amount (EUR)':                             'Total amount (EUR)',
    'Start date':                                    'Project start date',
    'Start Year':                                    'Year project started',
    'End Year':                                      'End date',
    'State of standardized research organization':   'PI organisation state',
    'Country of standardized research organization': 'PI organisation country',
    'Funder':                                        'Funder name',
    'Funder Country':                                'Funder Country',
    'Source Linkout':                                'URL for announcement',
}

# Per-stage definitions - what differs between rescat, end_product, award_purpose, and subpillar.
# 'pillar_split': whether the prompt/tool/category list varies by pillar (rescat, subpillar) or is
# shared across all pillars (end_product, award_purpose).
# 'classified_value_cols'/'curated_value_cols': parallel lists - the funding_classified diagnostic
# column(s) and the corresponding funding_curated target column(s), in the same order.
# 'derive_fn(tool_input, field_map, pillar)': turns a raw tool-call input into a tuple of values,
# one per classified_value_cols/curated_value_cols entry. 'pillar' is None for non-pillar-split
# stages, and for pillar-split stages whose derive logic doesn't depend on it (rescat).
STAGES = {
    'rescat': {
        'run_flag_name': 'RUN_RESCAT',
        'pillar_split': True,
        'valid_pillars': list(PILLAR_CATS.keys()),
        'prompt_paths': PROMPT_PATHS,
        'cats': PILLAR_CATS,
        'schema_mode': 'multi_label',
        'tool_name': 'label_research_category',
        'tool_description': 'Record TRUE/FALSE for each research category the grant substantively funds.',
        'classified_value_cols': ['research_category_LLM'],
        'classified_status_col': 'category_status_LLM',
        'classified_stop_reason_col': 'category_stop_reason_LLM',
        'curated_value_cols': ['Research category'],
        'promotes': True,
    },
    'end_product': {
        'run_flag_name': 'RUN_ENDPRODUCT',
        'pillar_split': False,
        'valid_pillars': None,
        'prompt_paths': PROMPT_PATH_ENDPRODUCT,
        'cats': ENDPRODUCT_CATS,
        'schema_mode': 'multi_label',
        'tool_name': 'label_end_product',
        'tool_description': 'Record TRUE/FALSE for each end product category the grant substantively targets.',
        'classified_value_cols': ['end_product_type_LLM', 'sub_end_product_LLM'],
        'classified_status_col': 'endproduct_status_LLM',
        'classified_stop_reason_col': 'endproduct_stop_reason_LLM',
        'curated_value_cols': ['End product type', 'sub-end product'],
        'promotes': False,
    },
    'award_purpose': {
        'run_flag_name': 'RUN_AWARDPURPOSE',
        'pillar_split': False,
        'valid_pillars': None,
        'prompt_paths': PROMPT_PATH_AWARDPURPOSE,
        'cats': AWARDPURPOSE_CATS,
        'schema_mode': 'multi_label',
        'tool_name': 'label_award_purpose',
        'tool_description': 'Record TRUE/FALSE for each award purpose category that applies to the grant.',
        'classified_value_cols': ['award_purpose_LLM'],
        'classified_status_col': 'awardpurpose_status_LLM',
        'classified_stop_reason_col': 'awardpurpose_stop_reason_LLM',
        'curated_value_cols': ['Award purpose'],
        'promotes': False,
    },
    'subpillar': {
        'run_flag_name': 'RUN_SUBPILLAR',
        'pillar_split': True,
        'valid_pillars': list(SUBPILLAR_CATS.keys()),
        'prompt_paths': SUBPILLAR_PROMPT_PATHS,
        'cats': SUBPILLAR_CATS,
        'schema_mode': SUBPILLAR_SCHEMA_MODE,
        'tool_name': 'label_subpillar',
        'tool_description': 'Record the subpillar classification for the grant.',
        'classified_value_cols': ['subpillar_LLM'],
        'classified_status_col': 'subpillar_status_LLM',
        'classified_stop_reason_col': 'subpillar_stop_reason_LLM',
        'curated_value_cols': ['Sub-production pillar'],
        'promotes': False,
    },
}


def _field_name(category):
    """Sanitize a category name into a valid tool-schema property name (e.g.
    'Food safety & quality' -> 'Food_safety_quality')."""
    return re.sub(r'\W+', '_', category).strip('_')


def _build_category_tool(categories, tool_name, description):
    field_map = {_field_name(c): c for c in categories}  # sanitized field name -> real category name
    return {
        "name": tool_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {fname: {"type": "boolean"} for fname in field_map},
            "required": list(field_map.keys()),
        }
    }, field_map


def _build_single_label_tool(categories, tool_name, description):
    return {
        "name": tool_name,
        "description": description,
        "input_schema": {
            "type": "object",
            "properties": {"category": {"type": "string", "enum": list(categories)}},
            "required": ["category"],
        }
    }, None


def _derive_research_category(tool_input, field_map, pillar):
    """Comma-joined string of every category flagged TRUE, suppressing 'Other' whenever at
    least one specific category is also TRUE (matches genai_rescat_testing.ipynb's validated logic)."""
    true_labels = [orig for fname, orig in field_map.items() if tool_input.get(fname) is True]
    if 'Other' in true_labels and len(true_labels) > 1:
        true_labels = [l for l in true_labels if l != 'Other']
    return (', '.join(true_labels) if true_labels else 'NA',)


def _derive_end_product(tool_input, field_map, pillar):
    """(end_product_type, sub_end_product) - dairy subcategories collapse to 'Dairy' in the
    primary field and list their specific names in the secondary field, matching
    grant_endproduct_testing.ipynb's validated build_predictions logic."""
    true_cats = [orig for fname, orig in field_map.items() if tool_input.get(fname) is True]
    # 'Agnostic' means "not aimed at a specific product" - combining it with a real product is
    # contradictory, so drop it whenever at least one specific category is also true (mirrors
    # _derive_research_category's identical 'Other' suppression above).
    if 'Agnostic' in true_cats and len(true_cats) > 1:
        true_cats = [c for c in true_cats if c != 'Agnostic']
    dairy_true = [c for c in true_cats if c in DAIRY_SUBCATS]
    non_dairy_true = [c for c in true_cats if c not in DAIRY_SUBCATS and c != 'Dairy']
    is_dairy = 'Dairy' in true_cats or bool(dairy_true)
    end_product_type = non_dairy_true + (['Dairy'] if is_dairy else [])
    return ('; '.join(end_product_type), '; '.join(dairy_true))


def _derive_award_purpose(tool_input, field_map, pillar):
    true_cats = [orig for fname, orig in field_map.items() if tool_input.get(fname) is True]
    return ('; '.join(true_cats),)


def _derive_subpillar(tool_input, field_map, pillar):
    """Matches grant_subpillar_testing.ipynb's validated collapse_fermentation/collapse_plantbased/
    collapse_crosscutting logic exactly, including collapse_fermentation's known gap: if neither BF
    nor PF is true, returns '' - relying entirely on the prompt's own "must resolve to something"
    instruction, same as the notebook."""
    if SUBPILLAR_SCHEMA_MODE[pillar] == 'single_label':
        return (tool_input.get('category', '') or '',)
    true_cats = [orig for fname, orig in field_map.items() if tool_input.get(fname) is True]
    if pillar == 'F':
        bf, pf = 'BF' in true_cats, 'PF' in true_cats
        if bf and pf:
            return ('Mixed',)
        if bf:
            return ('BF',)
        if pf:
            return ('PF',)
        return ('',)
    if pillar == 'PB':
        return ('TF' if 'Traditional fermentation' in true_cats else '',)
    return ('',)


STAGE_DERIVE_FNS = {
    'rescat':        _derive_research_category,
    'end_product':   _derive_end_product,
    'award_purpose': _derive_award_purpose,
    'subpillar':     _derive_subpillar,
}


def _split_first_rest(value):
    """Split a semicolon-joined value into (first entry, remaining entries joined) - mirrors
    Funding_dedup_helpers.gap_fill_researchers_and_orgs' splitting convention."""
    import pandas as pd
    if value is None or (not isinstance(value, str) and pd.isna(value)):
        return None, None
    parts = [p.strip() for p in str(value).split(';') if p.strip()]
    if not parts:
        return None, None
    return parts[0], ('; '.join(parts[1:]) or None)


def _resolve_stage_assets(stage):
    """{pillar (or '_' if not pillar-split): (system_prompt_text, tool_dict, field_map)}."""
    assets = {}
    if stage['pillar_split']:
        mode_by_pillar = stage['schema_mode'] if isinstance(stage['schema_mode'], dict) \
            else {p: stage['schema_mode'] for p in stage['valid_pillars']}
        for pillar in stage['valid_pillars']:
            with open(stage['prompt_paths'][pillar], 'r', encoding='utf-8') as f:
                text = f.read().strip()
            cats = stage['cats'][pillar]
            if mode_by_pillar[pillar] == 'single_label':
                tool, field_map = _build_single_label_tool(cats, stage['tool_name'], stage['tool_description'])
            else:
                tool, field_map = _build_category_tool(cats, stage['tool_name'], stage['tool_description'])
            assets[pillar] = (text, tool, field_map)
    else:
        with open(stage['prompt_paths'], 'r', encoding='utf-8') as f:
            text = f.read().strip()
        tool, field_map = _build_category_tool(stage['cats'], stage['tool_name'], stage['tool_description'])
        assets['_'] = (text, tool, field_map)
    return assets


def main(
    KEY_PATH=KEY_PATH,
    DB_PATH=DB_PATH,
    CLASSIFICATION_TABLE=CLASSIFICATION_TABLE,
    CURATED_TABLE=CURATED_TABLE,
    SCOPE_COL=SCOPE_COL,
    PILLAR_COL=PILLAR_COL,
    RUN_RESCAT=RUN_RESCAT,
    RUN_ENDPRODUCT=RUN_ENDPRODUCT,
    RUN_AWARDPURPOSE=RUN_AWARDPURPOSE,
    RUN_SUBPILLAR=RUN_SUBPILLAR,
    LABEL_SCOPE=LABEL_SCOPE,
    PROMPT_PATHS=PROMPT_PATHS,
    PROMPT_PATH_ENDPRODUCT=PROMPT_PATH_ENDPRODUCT,
    PROMPT_PATH_AWARDPURPOSE=PROMPT_PATH_AWARDPURPOSE,
    SUBPILLAR_PROMPT_PATHS=SUBPILLAR_PROMPT_PATHS,
    LLM_MODEL_LABEL=LLM_MODEL_LABEL,
    MAX_TOKENS=MAX_TOKENS,
    TEMPERATURE=TEMPERATURE,
    BATCH_DIR=BATCH_DIR,
    POLL_INTERVAL_SECONDS=POLL_INTERVAL_SECONDS,
    RUN_LABEL=RUN_LABEL,
    OUTPUT_DIR=OUTPUT_DIR,
):
    import time
    import json
    from datetime import datetime
    from pathlib import Path

    import anthropic
    import duckdb
    import pandas as pd
    from dotenv import load_dotenv

    run_flags = {
        'rescat': RUN_RESCAT, 'end_product': RUN_ENDPRODUCT,
        'award_purpose': RUN_AWARDPURPOSE, 'subpillar': RUN_SUBPILLAR,
    }
    # re-point each stage's dynamic bits (prompt paths passed as kwargs, not module globals) so a
    # caller overriding e.g. PROMPT_PATH_ENDPRODUCT actually takes effect
    prompt_paths_by_stage = {
        'rescat': PROMPT_PATHS, 'end_product': PROMPT_PATH_ENDPRODUCT,
        'award_purpose': PROMPT_PATH_AWARDPURPOSE, 'subpillar': SUBPILLAR_PROMPT_PATHS,
    }
    active_stages = {}
    for name, cfg in STAGES.items():
        if not run_flags[name]:
            continue
        stage = dict(cfg)
        stage['prompt_paths'] = prompt_paths_by_stage[name]
        stage['derive_fn'] = STAGE_DERIVE_FNS[name]
        active_stages[name] = stage

    if not active_stages:
        print("No stages enabled (all RUN_* flags are False). Nothing to do.")
        return

    # 1. Authenticate with the Anthropic API
    print("\nConnecting to the Anthropic API")

    load_dotenv(KEY_PATH)
    client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))

    batch_dir = Path(BATCH_DIR)
    batch_dir.mkdir(exist_ok=True)

    # 2. Resolve RUN_LABEL: resume the newest incomplete labelling batch, or mint a fresh one
    def find_incomplete_batch():
        files = sorted(
            (f for f in os.listdir(batch_dir) if f.endswith('_llm_labelling.json')),
            reverse=True,
        )
        for fname in files:
            meta = json.loads((batch_dir / fname).read_text(encoding="utf-8"))
            if 'completed_at' not in meta:
                return meta['run_label']
        return None

    if RUN_LABEL is None:
        RUN_LABEL = find_incomplete_batch()
    if RUN_LABEL is None:
        RUN_LABEL = f"labelling_{datetime.today().strftime('%y%m%d_%H%M')}"

    metadata_path = batch_dir / f"{RUN_LABEL}_llm_labelling.json"
    is_resuming = metadata_path.exists()

    if is_resuming:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        print(f"Found existing labelling run '{RUN_LABEL}'. Resuming without resubmitting already-submitted stages.")
    else:
        metadata = {"run_label": RUN_LABEL, "created_at": datetime.now().isoformat(), "stages": {}}

    db = duckdb.connect(database=DB_PATH)

    # 3. Load funding_classified + funding_curated once, shared across every stage
    fc = db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE}").df()
    curated_df = db.sql(f"SELECT * FROM {CURATED_TABLE}").df()

    # LABEL_SCOPE='all' overwrites existing labels, so take a one-time snapshot of
    # funding_curated exactly as it stood before this run touches anything - the only record of
    # that prior state, since S2's own date_curated column gets uniformly re-stamped (not
    # preserved per-row) every time funding_curated is rebuilt.
    if LABEL_SCOPE == 'all' and not is_resuming:
        snapshot_table = f"funding_curated_pre_relabel_{RUN_LABEL}"
        db.sql(f"CREATE OR REPLACE TABLE {snapshot_table} AS SELECT * FROM curated_df")
        metadata['pre_relabel_snapshot_table'] = snapshot_table
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"LABEL_SCOPE='all': saved a pre-relabelling snapshot of '{CURATED_TABLE}' to "
              f"'{snapshot_table}' ({len(curated_df)} rows) before any labels are overwritten.")

    curated_df['_ap_pillar_code'] = (
        curated_df['AP pillar'].astype(str).str.strip().str.lower().map(AP_PILLAR_TO_CODE)
    )
    already_promoted_ids = set(curated_df['Identification code'].dropna().astype(str))

    def build_pool_a(stage):
        mask = (fc[SCOPE_COL] == 'in') & (~fc['Grant ID'].astype(str).isin(already_promoted_ids))
        if stage['pillar_split']:
            mask &= fc[PILLAR_COL].isin(stage['valid_pillars'])
        status_col = stage['classified_status_col']
        if status_col not in fc.columns:
            fc[status_col] = None
        if LABEL_SCOPE == 'new_only':
            mask &= fc[status_col].isna()
        return fc[mask].reset_index(drop=True)

    def build_pool_b(stage):
        mask = pd.Series(True, index=curated_df.index)
        if stage['pillar_split']:
            mask &= curated_df['_ap_pillar_code'].isin(stage['valid_pillars'])
        value_col = stage['curated_value_cols'][0]
        if value_col not in curated_df.columns:
            curated_df[value_col] = None
        if LABEL_SCOPE == 'new_only':
            missing = curated_df[value_col].isna() | (curated_df[value_col].astype(str).str.strip() == '')
            mask &= missing
        return curated_df[mask].index.tolist()

    def build_request(custom_id, tool, prompt_text, title, abstract):
        user_message = f"Title: {title}\n\nAbstract: {abstract}"
        return {
            "custom_id": custom_id,
            "params": {
                "model": LLM_MODEL_LABEL,
                "max_tokens": MAX_TOKENS,
                "temperature": TEMPERATURE,
                "system": [{"type": "text", "text": prompt_text, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": user_message}],
                "tools": [tool],
                "tool_choice": {"type": "tool", "name": tool["name"]},
            }
        }

    # 4. Build pools + submit a batch per enabled stage that isn't already in the metadata
    stage_assets = {}
    for stage_name, stage in active_stages.items():
        stage_assets[stage_name] = _resolve_stage_assets(stage)

        for col in stage['curated_value_cols']:
            if col not in curated_df.columns:
                curated_df[col] = None

        if stage_name in metadata['stages']:
            print(f"[{stage_name}] Already submitted (batch {metadata['stages'][stage_name]['batch_id']}), skipping resubmission.")
            continue

        assets = stage_assets[stage_name]
        pool_a = build_pool_a(stage)
        pool_b_indices = build_pool_b(stage)
        print(f"[{stage_name}] Pool A (funding_classified): {len(pool_a)} rows")
        print(f"[{stage_name}] Pool B (funding_curated): {len(pool_b_indices)} rows")

        if len(pool_a) == 0 and len(pool_b_indices) == 0:
            print(f"[{stage_name}] No eligible rows. Skipping.")
            continue

        batch_requests = []
        pillar_by_custom_id = {}

        for _, row in pool_a.iterrows():
            pillar = row[PILLAR_COL] if stage['pillar_split'] else '_'
            text, tool, _ = assets[pillar]
            custom_id = f"fc_{row['Grant ID'].replace('.', '_')}"
            batch_requests.append(build_request(custom_id, tool, text, row['Title translated'], row['Abstract translated']))
            pillar_by_custom_id[custom_id] = pillar

        for idx in pool_b_indices:
            row = curated_df.loc[idx]
            pillar = curated_df.loc[idx, '_ap_pillar_code'] if stage['pillar_split'] else '_'
            text, tool, _ = assets[pillar]
            custom_id = f"curated_{idx}"
            batch_requests.append(build_request(custom_id, tool, text, row['Title'], row['Abstract']))
            pillar_by_custom_id[custom_id] = pillar

        print(f"[{stage_name}] Submitting batch of {len(batch_requests)} requests")
        batch = client.messages.batches.create(requests=batch_requests)
        print(f"[{stage_name}] Batch ID: {batch.id}   Status: {batch.processing_status}")

        metadata['stages'][stage_name] = {
            "batch_id": batch.id,
            "pillar_by_custom_id": pillar_by_custom_id,
            "n_records": len(batch_requests),
        }
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    if not metadata['stages']:
        print("\nNo batches were submitted for any enabled stage (nothing eligible). Nothing to do.")
        db.close()
        return

    # 5. One shared polling loop across every stage's outstanding batch
    print("\nWaiting for all submitted batches to complete")

    pending = set(metadata['stages'].keys())
    while pending:
        for stage_name in list(pending):
            batch = client.messages.batches.retrieve(metadata['stages'][stage_name]['batch_id'])
            print(f"[{stage_name}] status: {batch.processing_status}   counts: {batch.request_counts}")
            if batch.processing_status == "ended":
                pending.discard(stage_name)
        if pending:
            time.sleep(POLL_INTERVAL_SECONDS)

    # 6. Retrieve + parse each stage's results, writing Pool A onto funding_classified and
    # Pool B directly onto the in-memory curated_df
    print("\nRetrieving results")

    date_labelling = datetime.today().strftime('%y%m%d')
    all_failures = []  # every non-'ok' result across every stage/pool, for the end-of-run export

    for stage_name, stage in active_stages.items():
        if stage_name not in metadata['stages']:
            continue  # nothing was eligible/submitted for this stage
        batch_id = metadata['stages'][stage_name]['batch_id']
        pillar_by_custom_id = metadata['stages'][stage_name]['pillar_by_custom_id']
        assets = stage_assets[stage_name]

        # force object dtype before writing string labels into curated_df - if a target column is
        # entirely null (e.g. never populated, or - as in a small test slice - coincidentally all
        # blank), DuckDB/pandas can infer a numeric dtype for it, which then rejects a string write
        for col in stage['curated_value_cols']:
            curated_df[col] = curated_df[col].astype(object)

        fc_results = []
        n_curated_total = 0
        n_curated_ok = 0

        for result in client.messages.batches.results(batch_id):
            custom_id = result.custom_id
            pillar = pillar_by_custom_id.get(custom_id, '_')
            _, _, field_map = assets.get(pillar, (None, None, None))

            if result.result.type == "succeeded":
                content = result.result.message.content
                stop_reason = result.result.message.stop_reason
                tool_block = next((b for b in content if b.type == "tool_use"), None)
                if tool_block:
                    values = stage['derive_fn'](tool_block.input, field_map, pillar if stage['pillar_split'] else None)
                    status = "ok"
                else:
                    values = tuple(None for _ in stage['classified_value_cols'])
                    status = "parse_error"
            else:
                stop_reason = None
                values = tuple(None for _ in stage['classified_value_cols'])
                status = result.result.type

            if custom_id.startswith("fc_"):
                grant_id = custom_id[len("fc_"):].replace("_", ".")
                row = {
                    "Grant ID": grant_id,
                    stage['classified_status_col']: status,
                    stage['classified_stop_reason_col']: stop_reason,
                    "date_labelling": date_labelling,
                }
                for col, val in zip(stage['classified_value_cols'], values):
                    row[col] = val
                fc_results.append(row)
                if status != "ok":
                    all_failures.append({
                        "pool": "A", "stage": stage_name, "grant_id": grant_id,
                        "identification_code": None, "title": None, "curated_row_index": None,
                        "status": status, "stop_reason": stop_reason,
                    })
            elif custom_id.startswith("curated_"):
                idx = int(custom_id[len("curated_"):])
                n_curated_total += 1
                if status == "ok":
                    for col, val in zip(stage['curated_value_cols'], values):
                        curated_df.at[idx, col] = val
                    n_curated_ok += 1
                else:
                    all_failures.append({
                        "pool": "B", "stage": stage_name, "grant_id": None,
                        "identification_code": curated_df.at[idx, 'Identification code'],
                        "title": curated_df.at[idx, 'Title'], "curated_row_index": idx,
                        "status": status, "stop_reason": stop_reason,
                    })

        fc_results_df = pd.DataFrame(fc_results, columns=[
            "Grant ID", *stage['classified_value_cols'], stage['classified_status_col'],
            stage['classified_stop_reason_col'], "date_labelling",
        ])
        n_ok = (fc_results_df[stage['classified_status_col']] == 'ok').sum() if len(fc_results_df) else 0
        print(f"[{stage_name}] Pool A results: {len(fc_results_df)} total, {n_ok} succeeded")
        print(f"[{stage_name}] Pool B results: {n_curated_total} total, {n_curated_ok} succeeded")

        if len(fc_results_df):
            llm_columns = {col: 'VARCHAR' for col in [
                *stage['classified_value_cols'], stage['classified_status_col'],
                stage['classified_stop_reason_col'], 'date_labelling',
            ]}
            for col, dtype in llm_columns.items():
                db.sql(f'ALTER TABLE {CLASSIFICATION_TABLE} ADD COLUMN IF NOT EXISTS {col} {dtype}')

            db.register(f'{stage_name}_results', fc_results_df)
            set_clause = ", ".join(f"{col} = {stage_name}_results.{col}" for col in llm_columns)
            db.sql(f"""
                UPDATE {CLASSIFICATION_TABLE}
                SET {set_clause}
                FROM {stage_name}_results
                WHERE {CLASSIFICATION_TABLE}."Grant ID" = {stage_name}_results."Grant ID"
            """)
            print(f"[{stage_name}] '{CLASSIFICATION_TABLE}' updated with diagnostic columns for {len(fc_results_df)} rows.")

    # 7. Promote not-yet-promoted, successfully rescat-labelled Pool A rows into funding_curated,
    # carrying over whatever the other enabled stages have already computed for that same grant
    if 'rescat' in active_stages and 'rescat' in metadata['stages']:
        fc_full = db.sql(f"SELECT * FROM {CLASSIFICATION_TABLE}").df()
        rescat_stage = active_stages['rescat']
        # Promote anything rescat *attempted*, not just what succeeded - a failed rescat call
        # (parse_error/errored/etc.) still writes a real (blank) value into the promoted row,
        # and build_pool_b's blank-value check will pick it up and retry automatically on a
        # future run. Gating on == 'ok' instead would leave failures stuck in funding_classified
        # forever, since Pool A's new_only filter also excludes any row with a non-null status.
        to_promote = fc_full[
            (fc_full[rescat_stage['classified_status_col']].notna()) &
            (~fc_full['Grant ID'].astype(str).isin(already_promoted_ids))
        ]

        if len(to_promote):
            new_rows = []
            for _, row in to_promote.iterrows():
                new_row = {lrd_col: row.get(src_col) for src_col, lrd_col in PROMOTE_COL_MAP.items()}

                names_first, names_rest = _split_first_rest(row.get('Researchers'))
                if names_first:
                    new_row['Project lead (PI)'] = names_first
                    if names_rest:
                        new_row['Collaborator names'] = names_rest

                org_first, org_rest = _split_first_rest(row.get('Research Organization - standardized'))
                if org_first:
                    new_row['PI organisation'] = org_first
                    if org_rest:
                        new_row['Collaborator institutions'] = org_rest

                new_row['Database'] = 'Dimensions'
                new_row['Identification code'] = row['Grant ID']
                new_row['AP pillar'] = CODE_TO_AP_PILLAR.get(row.get(PILLAR_COL))
                new_row['Research category'] = row.get('research_category_LLM')

                # carry over any other enabled stage's already-computed labels for this grant, so a
                # brand-new grant gets all four labels in the same run it's promoted
                for other_name, other_stage in active_stages.items():
                    if other_name == 'rescat':
                        continue
                    for classified_col, curated_col in zip(other_stage['classified_value_cols'], other_stage['curated_value_cols']):
                        new_row[curated_col] = row.get(classified_col)

                new_rows.append(new_row)

            new_rows_df = pd.DataFrame(new_rows).reindex(columns=[c for c in curated_df.columns if c != '_ap_pillar_code'])
            curated_df = pd.concat([curated_df, new_rows_df], ignore_index=True)
            print(f"Promoted {len(new_rows_df)} newly-labelled Dimensions grants into '{CURATED_TABLE}'.")
        else:
            print("No new Dimensions grants to promote.")
    else:
        print("Rescat stage did not run this session - skipping promotion "
              "(any Pool A rows stay staged in funding_classified until a future run promotes them).")

    # 8. Persist funding_curated once, atomically, with every stage's Pool B updates and any
    # newly-promoted rows
    curated_df = curated_df.drop(columns=['_ap_pillar_code'], errors='ignore')
    db.sql(f"CREATE OR REPLACE TABLE {CURATED_TABLE} AS SELECT * FROM curated_df")
    print(f"'{CURATED_TABLE}' rewritten with {len(curated_df)} total rows.")

    # 9. Excel copy of the final, updated funding_curated dataset (RUN_LABEL leading the
    # filename, so reruns don't overwrite an earlier export, and it's traceable to this run)
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)
    export_path = output_dir / f"{RUN_LABEL}_{CURATED_TABLE}.xlsx"
    curated_df.to_excel(export_path, index=False)
    print(f"Excel copy of '{CURATED_TABLE}' saved to {export_path}")

    # Combined failures export (both pools, every enabled stage) - so a failure can be found and
    # directly fixed/retried without hunting through funding_classified/funding_curated by hand.
    # Pool A failures will self-retry automatically on a future new_only run (status column is
    # non-null but the row was never promoted); Pool B failures self-retry too, since the target
    # curated_value_col was left blank on failure and build_pool_b's new_only check picks up any
    # blank value regardless of cause - this export is for visibility/manual triage, not required
    # for correctness.
    if all_failures:
        failures_df = pd.DataFrame(all_failures)
        failures_path = output_dir / f"{RUN_LABEL}_labelling_failures.xlsx"
        failures_df.to_excel(failures_path, index=False)
        print(f"{len(failures_df)} failed labelling attempt(s) across all stages - saved to {failures_path}")
    else:
        print("No failed labelling attempts this run.")

    # Excel copy of this run's pre-relabelling snapshot, if one was taken (LABEL_SCOPE='all')
    snapshot_table = metadata.get('pre_relabel_snapshot_table')
    if snapshot_table:
        snapshot_df = db.sql(f"SELECT * FROM {snapshot_table}").df()
        snapshot_export_path = output_dir / f"{RUN_LABEL}_funding_curated_pre_relabel_snapshot.xlsx"
        snapshot_df.to_excel(snapshot_export_path, index=False)
        print(f"Excel copy of the pre-relabelling snapshot saved to {snapshot_export_path}")

    metadata["completed_at"] = datetime.now().isoformat()
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    db.close()
    print("\nDone!")


if __name__ == '__main__':
    main()
