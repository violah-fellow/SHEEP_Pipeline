# Funding Pipeline Manual

For running the automated grants/funding data collection, deduplication, scoping, and labelling pipeline (`Funding/`).

This pipeline shares its API keys, conda environment, and general repo setup with the Publications and Patents pipelines — see the combined **Publications/Patents step-by-step manual** for all of that. This document only covers what's different about Funding, which is a lot: there is no ML classification step, every deduplicated grant goes straight to an LLM, and there are roughly **nine separate manual review checkpoints** spread across three notebooks. If you've only ever run Publications or Patents, read this whole document before your first run — the shape of the pipeline is genuinely different, not just a renamed copy.

---

## Part 1: Prerequisites

Do the one-time setup from the combined manual's **Part 1** first (Miniconda, Git, VS Code, cloning the repo, `conda env create -f env.yaml`, `.env` file). Funding uses the exact same `sheep` conda environment and the exact same root-level `.env` file — there is nothing Funding-specific to install there.

Two things ARE specific to Funding and aren't covered by the combined manual:

**1. Two source spreadsheets must exist before you can run Step 2 (deduplication) for the first time.** Unlike Publications/Patents, Funding merges in data that doesn't come from Dimensions at all:
- `Funding/raw_data/GrantsTracker_<date>.xlsx` — an export of GFI's Grants Tracker (Airtable).
- `Funding/raw_data/Funding<year>_inscope.xlsx` — last report's curated/in-scope export (i.e. the previous cycle's `funding_curated` output).

These are supplied by hand — nothing in the pipeline generates or downloads them. `S2_grant_deduplication.ipynb` will fail at its load cell if they're missing or misnamed. Check the `LAST_REPORT_FILE` and `GRANTS_TRACKER_FILE` variables at the top of that notebook match whatever you actually placed in `raw_data/`.

**2. No `embeddings/` or `models/` folder.** Publications and Patents both have an ML classification gate (a trained logistic-regression model, `SCOPE_MODEL_PATH`/`PILLAR_MODEL_PATH`/`THRESHOLD_PATH`, plus embeddings on disk) that filters results before they ever reach the LLM. Funding has none of this — `S3_LLM_scope.py`'s own header comment says it explicitly: *"Unlike Publications, there is no ML gate here — every row in DEDUP_TABLE is sent to the LLM."* Don't go looking for an `embeddings/` folder or model files for Funding; they don't exist by design.

### Folders — what you need to create vs. what's automatic

| Folder | Created automatically? | By what |
|---|---|---|
| `status_logs/`, `run_logs/` | Yes | `pipeline_funding.py` on any run (`os.makedirs(..., exist_ok=True)`) |
| `batch_jobs/` | Yes | `S3_LLM_scope.py` / `S6_LLM_labelling.py` the first time either submits a batch |
| `data_review/`, `data_output/` | Yes | `S6_LLM_labelling.py`, on demand |
| `data_audit/` | Yes | `S2_grant_deduplication.ipynb` (`AUDIT_DIR.mkdir(exist_ok=True)`) and `export_combined_report.py` (`os.makedirs(OUT_DIR, exist_ok=True)`) |
| `raw_data/` | **No — create by hand** | You must create this and place the two source spreadsheets in it |
| `data_analysis/` | Not used by any current script | Old leftover files only — no active code writes here (see Part 12) |

---

## Part 2: Overall Pipeline Flow

```
S1 (query, automated)
   │  Query Dimensions for new grants → staging table run_<timestamp>
   ▼
S2 (dedup + 3-way merge, MANUAL NOTEBOOK — hard stop)
   │  Merge Dimensions + last report + Grants Tracker, dedupe, rebuild funding_curated
   │  → run_<timestamp>_dedup (genuinely-new grants only, feeds S3)
   ▼
S3 (LLM scope, automated/batch)
   │  Claude decides in/out of scope + pillar flags for every row in the dedup table
   │  → writes scope_LLM/pillar_LLM/confidence_LLM etc. onto funding_classified
   ▼
S5 (scope/pillar review, MANUAL NOTEBOOK — hard stop)
   │  Auto-accepts high-confidence rows, exports the rest for human scope_curated/pillar_curated
   ▼
S6 (LLM labelling ×4 stages + promotion, automated/batch)
   │  Research category, End product, Award purpose, Subpillar — then promotes newly-labelled
   │  Dimensions grants into funding_curated (rewritten CREATE OR REPLACE each run)
   ▼
S7 (currency backfill + QA, MANUAL/standalone — not run by pipeline_funding.py)
   │  Backfills missing currency conversions, splits funding per-year, flags label gaps
   ▼
Funding_analysis.ipynb (reporting, standalone)
```

**Important structural point:** `pipeline_funding.py` only calls **S1, S3, and S6** as Python functions. **S2 and S5 are Jupyter notebooks** that the orchestrator does not and cannot run for you — it detects whether their expected output already exists in the database and either continues past them or prints instructions and calls `sys.exit(0)` to stop the whole script until you've done the manual work. **S7 and `Funding_analysis.ipynb` are entirely outside the orchestrator** — run them by hand whenever you need to, there's no "step" for them in `status_logs/`.

You'll also notice there's no `S4_*.py` file. The pipeline's five internal step names (`query`, `dedup`, `llm_scope`, `review`, `llm_labelling`) map to files S1→S2→S3→S5→S6 — S5 fills the role a hypothetical "S4" might otherwise have had. This is intentional (it mirrors the sibling pipelines' file layout) — not a missing file, nothing to fix.

Because of the two hard stops, **treat a full funding run as a multi-session, stop-and-resume process**, not a single unattended overnight run like Publications/Patents. Expect to: run the script, do S2 by hand, re-run the script, do S5 by hand, re-run the script to finish.

---

## Part 3: File Structure

```
Funding/
├── pipeline_funding.py              orchestrator: calls S1 → S3 → S6, gates on S2/S5
├── S1_query_dimensions.py           query Dimensions API → staging table
├── S2_grant_deduplication.ipynb     MANUAL: 3-way merge + dedup, rebuilds funding_curated
├── S3_LLM_scope.py                  LLM in/out-of-scope + pillar classification (batch)
├── S5_Review_classifications.ipynb  MANUAL: auto-curate + human review of scope/pillar
├── S6_LLM_labelling.py              LLM labelling (4 stages) + promotion to funding_curated
├── S7_data_transformation.ipynb     MANUAL/standalone: currency backfill, per-year split, QA
├── Funding_analysis.ipynb           standalone: final reporting
├── Funding_analysis_helpers.py      helpers for the analysis notebook
├── Funding_dedup_helpers.py         shared helpers: country/region cleanup, gap-fill, review
│                                     export/read-back — used by S2, S6, S7
├── Helper_pipeline_functions.py     status-log helpers used only by pipeline_funding.py
├── dimensions_search_funding.txt    Dimensions search strings, one per line
├── funding.db (+ .db.wal)           DuckDB database — self-contained to Funding/
├── llm_prompts/                     system prompts for every LLM stage (see Part 9)
├── raw_data/          [create by hand]  source spreadsheets (Grants Tracker, last report)
├── data_audit/         [auto-created]    before/after highlighted-diff Excel audit trails
├── data_analysis/      [not used by code]  legacy files only, no active script writes here
├── data_review/        [auto-created]    manual-review CSV exports / reviewed files
├── data_output/        [auto-created]    final Excel exports (funding_curated, failures, etc.)
├── batch_jobs/         [auto-created]    Anthropic Batch API submission metadata
├── status_logs/        [auto-created]    checkpoint JSON, one per pipeline run
├── run_logs/           [auto-created]    tee'd console output, one .log per pipeline run
│
├── export_combined_report.py               one-off script, not part of S1-S7
├── patch_funding_curated_end_product_type.py   one-off patch script, not part of S1-S7
├── ap_pillar_backfill.ipynb                one-off backfill notebook, not part of S1-S7
└── funding_curated_gap_backfill.ipynb      one-off backfill notebook, not part of S1-S7
```

The four "one-off" files at the bottom are maintenance scripts that were run directly against `funding.db` outside the normal S1→S6 flow (you can tell because their outputs in `batch_jobs/`/`data_output/` use custom `RUN_LABEL`s like `ap_pillar_backfill` rather than a `run_YYMMDD_HHMM` timestamp). They're useful references for one-off fixes, but they are **not** part of the regular pipeline and have no corresponding entry in `status_logs/`.

---

## Part 4: `.gitignore` and `.env` explained

### `.env` — identical convention to Publications/Patents

There is **one** `.env` file, at the repo root (`SHEEP_Pipeline/.env`), shared by all three pipelines. It holds exactly two keys (see `.env.example` at the repo root):
```
DIMENSIONS_API_KEY=your_dimensions_key_here
CLAUDE_API_KEY=your_claude_api_key_here
```
Every Funding script that needs a key (`S1_query_dimensions.py`, `S3_LLM_scope.py`, `S6_LLM_labelling.py`, `pipeline_funding.py`) sets `KEY_PATH = '../.env'` and loads it with `python-dotenv` — the `'../.env'` is relative to the `Funding/` working directory, so it resolves back to the repo root. You never need a second `.env` inside `Funding/`.

### `.gitignore` — Funding-relevant entries

```
Funding/batch_jobs/*
Funding/status_logs/*
Funding/run_logs/*
Funding/__pycache__/*
Funding/funding.db
Funding/funding.db.wal
Funding/data_audit/*
Funding/data_review/*
Funding/data_output/*
Funding/raw_data/*
```

| Ignored path | Why |
|---|---|
| `funding.db`, `funding.db.wal` | The database is a large, regenerable binary — never commit it. |
| `batch_jobs/*`, `status_logs/*`, `run_logs/*` | Per-run runtime state/logs — regenerated every run, not meant to be shared via git. |
| `data_audit/*`, `data_review/*`, `data_output/*` | Derived/exported spreadsheets (audit diffs, review exports, final curated Excel copies) — large, regenerable from the DB, and often contain data you don't want to accidentally push. |
| `raw_data/*` | The manually-sourced Grants Tracker and last-report spreadsheets — these are inputs you place by hand each cycle, not something to version. |

**Not ignored (deliberately tracked in git), worth knowing:**
- `Funding/llm_prompts/*` — these are source-controlled prompt text, the same way code is. Editing a prompt is a real change you want in git history.
- `Funding/data_analysis/*` — the two `.xlsx` meeting-report snapshots currently in this folder are tracked. If this folder is going to keep accumulating one-off report exports, it may be worth adding it to `.gitignore` too (see Part 12) — right now it's tracked seemingly by omission rather than deliberate choice.

---

## Part 5: The Three Datasets and Their Column Maps

Funding merges **three sources**, each with its own column-naming convention, into one common "curated tracker" schema:

- **(A) Dimensions.ai API** — queried live in S1, DSL field names (`id`, `title`, `funding_usd`, …).
- **(B) Last report** — the prior cycle's curated export (`raw_data/Funding<year>_inscope.xlsx`), already in the tracker schema.
- **(C) GFI Grants Tracker** — an Airtable export (`raw_data/GrantsTracker_*.xlsx`), with `EXT_`-prefixed (external-facing) and `INT_`-prefixed (internal-only) columns.

`S2_grant_deduplication.ipynb` converts (A) into the tracker schema via an adapter cell, then reconciles all three into `funding_curated`.

| Concept | (A) Dimensions raw (DSL) | (A) after S2 adapter | (B) Last report / `funding_curated` | (C) Grants Tracker |
|---|---|---|---|---|
| Grant ID | `id` (`grant.NNNNNNNN`) | `Grant ID` | `Identification code` (+ `Legacy identification code`) | `Dimensions.ai grant ID` |
| Title (translated) | `title` | `Title translated` | `Title` | `EXT_Title` |
| Title (original) | `original_title` | `Title` | `Original title` | *(none)* |
| Abstract | `abstract` | `Abstract translated` | `Abstract` | `EXT_Abstract` |
| Funder name | `funder_org_name` | `Funder` | `Funder name` | `EXT_Funder name` |
| Funder country | `funder_org_countries` | `Funder Country` | `Funder Country` | `EXT_Funder country` |
| Funder region | *(derived)* | *(derived)* | `Funder region` | `EXT_Funder region` |
| Funder type | *(no signal)* | *(no signal)* | `Funder type` | `EXT_Funder type` |
| Currency (native) | `funding_currency` | `Currency` | `Currency` | `INT_Currency type` |
| Total amount (native) | reconstructed from per-currency fields | `Total amount` | `Total amount` | `INT_Total amount (actual currency)` |
| Total amount (USD) | `funding_usd` | `Total amount (USD)` | `Total amount (USD)` | `EXT_Total amount (USD)` |
| Total amount (EUR) | `funding_eur` | `Total amount (EUR)` | `Total amount (EUR)` | *(none — filled by S7 FX)* |
| Gov & NP contribution (native) | *(Total copied wholesale)* | — | `Gov & NP contribution` | `INT_Gov contribution (actual currency)` |
| Gov & NP contribution (USD) | — | — | `Gov & NP contribution (USD)` | `EXT_Gov contribution (USD)` |
| Gov & NP contribution (EUR) | — | — | `Gov & NP contribution (EUR)` | *(none — filled by S7 FX)* |
| Start date | `start_date` | `Start date` | `Project start date` | `EXT_Project start date (estimated)` |
| Start year | `start_year` | `Start Year` | `Year project started` | `EXT_Year project starts` |
| End year | `end_date` | `End Year` | `End date` | `INT_End Date` |
| Duration / years active | *(n/a)* | *(n/a)* | `duration (years)` / `Years active` (derived) | `EXT_Duration of award in years` / derived |
| PI / researchers | `investigators` (has role) / `researchers` | `Researchers` (PI-first, semicolon-joined) | `Project lead (PI)` + `Collaborator names` | `EXT_Project lead (PI)` + `EXT_Collaborator names` |
| Research org | `research_org_names` | `Research Organization - standardized` | `PI organisation` + `Collaborator institutions` | `EXT_PI organization` + `EXT_Collaborator organizations` |
| Research org country | `research_org_countries` | `Country of standardized research organization` | `PI organisation country` | `EXT_PI organization country` |
| Research org region | *(n/a)* | *(n/a)* | `PI organisation region` (derived) | `EXT_PI organization region` |
| Announcement URL | `linkout` (first entry) | `Source Linkout` | `URL for announcement` | `EXT_URL for announcement` |
| AP pillar | *(from S6 promotion)* | — | `AP pillar` | `EXT_Production platform` |
| Sub-production pillar | *(from S6 subpillar stage)* | — | `Sub-production pillar` | *(source column, if present)* |
| End product type | *(from S6)* | — | `End product type` + `sub-end product` | `EXT_End product type` |
| Award purpose | *(from S6)* | — | `Award purpose` | `EXT_Award purpose` |
| Research category | *(from S6)* | — | `Research category` | *(none — GFI-internal, not exported by Airtable)* |
| Funding decision | *(always "Awarded")* | — | `Funding decision` | `EXT_Funding decision` |
| Date added / modified | *(n/a)* | — | `Date added` / `Last modified` | `EXT_Date added` / `EXT_Last modified` |
| GFI-internal flags | *(n/a)* | — | `GFI grantee`, `GFI LOS`, `Link to LOS`, `GFI partner`, `Tier` | `INT_GFI grantee?`, `INT_GFI Los?`, `INT_Link to LoS`, `INT_GFI partner?`, `INT_Tier` |
| Per-year money columns | *(n/a — S7 derives)* | — | `2020`…`2035` (populated by S7's even-split across `Years active`) | `EXT_<year> expenditures` / `INT_...(#)` |
| Row surrogate keys | — | — | `lrd_row_id` | `gt_row_id` |

### How the three sources are joined/deduped (S2's 7 passes, in order)

| # | Pass | Match type | Human review needed? |
|---|---|---|---|
| 1 | Dimensions ↔ last report | exact `Grant ID` == `Identification code` | No |
| 2 | Dimensions ↔ Grants Tracker | exact `Grant ID` == `Dimensions.ai grant ID` | No |
| 3 | Grants Tracker ↔ last report | exact `Dimensions.ai grant ID` == `Identification code` (regex-guarded to `grant.NNNNNNNN`) | No |
| 4 | Grants Tracker ↔ last report | exact normalized title match | **Yes** |
| 5 | Grants Tracker ↔ last report | fuzzy title match (rapidfuzz ≥ 85) | **Yes** |
| 6 | Dimensions ↔ last report (+ leftover GT rows) | exact normalized title match | **Yes** |
| 7 | Dimensions ↔ last report | fuzzy title match (rapidfuzz ≥ 85) | **Yes** |

Whatever survives all 7 passes is genuinely new and becomes the `DEDUP_TABLE` (`<RUN_TABLE>_dedup`) that S3 picks up.

---

## Part 6: CONFIG Variables to Review Before Running

Open `pipeline_funding.py` and check the CONFIG section at the top.

> **Read this before touching anything:** these CONFIG values are only used when a run **starts**. The moment Step 1 finishes, they're snapshotted into `status_logs/status_<RUN_TABLE>.json`, and every later invocation for that run reads from that JSON — not from this file. Editing CONFIG here does nothing for a run already in progress. To change a setting mid-run, edit the JSON directly, or delete it to restart that run from scratch.

**Always review before starting a new run**
| Variable | Default | Notes |
|---|---|---|
| `START_YEAR` / `END_YEAR` | `2020` / `2025` | Range of grant start years to query (inclusive). Set equal for a single year. |
| `LLM_MODEL_SCOPE` | `'claude-sonnet-4-6'` | Or `'claude-haiku-4-5'` for a cheap test run. |
| `LLM_MODEL_LABEL` | `'claude-sonnet-4-6'` | Same idea, used for all 4 of S6's labelling stages. |
| `RUN_RESCAT` / `RUN_ENDPRODUCT` / `RUN_AWARDPURPOSE` / `RUN_SUBPILLAR` | all `True` | Which of S6's 4 labelling stages to run this time. |
| `LABEL_SCOPE` | `'new_only'` | `'new_only'` labels/promotes only rows missing a label; `'all'` relabels and overwrites every eligible row's existing labels. Only set to `'all'` deliberately (e.g. after changing a prompt and wanting to relabel everything) — it also snapshots a `funding_curated_pre_relabel_<RUN_LABEL>` table/Excel first, since a full relabel overwrites `date_curated` on every row, so use that snapshot if you need to see what changed. |

**Only change if the pipeline structure has changed**
| Variable | Default | Notes |
|---|---|---|
| `PROMPT_PATH` | `'llm_prompts/scope_prompt_funding.md'` | S3's system prompt. |
| `STRINGS_FILE` | `'dimensions_search_funding.txt'` | Dimensions search strings, one per line. |
| `CLASSIFICATION_TABLE` | `'funding_classified'` | The pipeline-owned table that accumulates across every run. |

**Most likely no change needed**
| Variable | Default | Notes |
|---|---|---|
| `KEY_PATH` | `'../.env'` | Points at the repo-root `.env`. |
| `DB_PATH` | `'funding.db'` | Self-contained to `Funding/`. |
| `RESUME` | `True` | Whether to look for and continue an incomplete run on startup. |
| `STATUS_DIR` / `LOG_DIR` | `'status_logs'` / `'run_logs'` | |

S6's own per-stage category lists, prompt paths, and retry limits (`MAX_RETRY_ATTEMPTS`, `MAX_TOKENS`, etc.) are **not** exposed in `pipeline_funding.py`'s CONFIG at all — they're static enough that you'd edit `S6_LLM_labelling.py` directly if you ever needed to change them.

Same reminder as the combined manual: **check your Claude API account has enough balance before running.** If it runs out mid-batch, rows will come back with missing/`None` values rather than an error, and they'll all end up funnelled into manual review.

---

## Part 7: Running the Pipeline

Use the combined manual's `screen` (macOS) / WSL2 (Windows) guidance for keeping a long-running process alive. The difference for Funding: **expect to run the script three separate times per cycle**, not once:

```
conda activate sheep
cd Funding
python pipeline_funding.py
```

1. **First run** — executes Step 1 (query), then stops at Step 2 with printed instructions to run `S2_grant_deduplication.ipynb`.
2. Open and run `S2_grant_deduplication.ipynb` top to bottom (working through its 6 manual review points — see Part 8). Confirm it wrote a table named `<RUN_TABLE>_dedup` into `funding.db`.
3. **Second run** — `python pipeline_funding.py` again. It detects the dedup table exists, marks Step 2 done, runs Step 3 (LLM scope, may take a while as it polls the Batch API every 10 minutes), then stops at Step 4 with instructions to run `S5_Review_classifications.ipynb`.
4. Open and run `S5_Review_classifications.ipynb` (see Part 8). Confirm every grant from this run has a `scope_curated` decision.
5. **Third run** — `python pipeline_funding.py` again. It detects the review is complete, marks Step 4 done, and runs Step 5 (LLM labelling + promotion). When it finishes, you'll see `Pipeline complete for run '<RUN_TABLE>'.` followed by a boxed **"ACTION REQUIRED: run S7_data_transformation.ipynb now"** reminder — the pipeline being "complete" only means S1/S3/S6 have run; `funding_curated` isn't actually finished until S7's currency backfill and QA checks have been done. S7 isn't gated the way S2/S5 are (nothing downstream in this script depends on it, so there's no hard stop) — it's a reminder, not a checkpoint, so don't skip it just because the script didn't force you to.

**Resuming after an interruption:** just re-run `python pipeline_funding.py` — `RESUME=True` by default, so it finds the most recent incomplete run in `status_logs/` and continues from wherever it left off. **Restarting a run from scratch:** delete its `status_logs/status_<RUN_TABLE>.json` file (this does *not* clean up the corresponding DB tables — they're just orphaned).

Afterwards, run `S7_data_transformation.ipynb` by hand (currency backfill + QA — see Part 9) and, when you want the latest numbers, `Funding_analysis.ipynb`.

---

## Part 8: Manual Review Deep-Dive

This is the part that differs most from Publications/Patents — there are review checkpoints in three different notebooks. For every one of them, the pattern is the same: **the notebook cell that reads your reviewed file back in takes a literal filename you type in** — it is not a wildcard/glob lookup. So an exact filename match matters, and on a case-sensitive filesystem (macOS/Linux) getting the case wrong will throw a `FileNotFoundError`; on Windows (case-insensitive) it'll happen to work anyway, which can mask the mistake until someone else on a different OS hits it. **The suffix convention is not consistent across review points** — check the exact suffix in the table below for whichever review point you're doing, don't assume they're all the same.

### S2 — deduplication notebook (6 review points)

| Review point | Exported file | Fill in | Save reviewed file as | Gotchas |
|---|---|---|---|---|
| Grants Tracker funder type | `data_review/<ts>_gt_funder_type_for_review.csv` | `EXT_Funder type` (e.g. Government, Nonprofit, For profit) | `<ts>_gt_funder_type_for_review_REVIEWED.csv` (**uppercase** `_REVIEWED`) | Leave blank if truly unknown. |
| GT ↔ last report, exact title match | `data_review/<ts>_gt_title_match_for_review.csv` | `is_true_match` (pre-filled `True`) — flip to `False` for any pair with a different start date, funding amount, Dimensions ID, or other distinguishing feature | `<ts>_gt_title_match_reviewed.csv` (**lowercase** `_reviewed`) | |
| GT ↔ last report, fuzzy title match | `data_review/<ts>_gt_partial_title_match_for_review.csv` | same `is_true_match` pattern | `<ts>_gt_partial_title_match_reviewed.csv` (lowercase) | Confirmed matches just remove the duplicate — no gap-fill, too uncertain. |
| Dimensions ↔ last report, exact title match | `data_review/<ts>_dim_title_match_for_review.csv` | `is_true_match` | `<ts>_dim_title_match_reviewed.csv` (lowercase) | **Must save as UTF-8** — the notebook says so explicitly. Confirmed matches overwrite `Identification code` with the Dimensions ID (old value preserved to `Legacy identification code`). |
| Dimensions ↔ last report, fuzzy title match | `data_review/<ts>_dim_partial_title_match_for_review.csv` | `is_true_match` | `<ts>_dim_partial_title_match_reviewed.csv` (lowercase) | A diagnostic cell prints match counts at several thresholds/scorers first to help sanity-check the fuzzy match — there's an open, unresolved discrepancy noted in the notebook between this pass's count and a prior manual pass's count (see Part 12). |
| AP pillar normalization | `data_review/<ts>_ap_pillar_for_review.csv` | `corrected_ap_pillar` — one of Plant-based / Fermentation / Cultivated / Cross-cutting | `<ts>_ap_pillar_reviewed.csv` (lowercase) | Only rows `normalize_ap_pillar()` couldn't auto-resolve show up here (casing/suffix variants and comma-joined multi-pillar values are handled automatically). |

If a reviewed file doesn't match the exact expected filename/casing, or is missing a row the notebook expects, `apply_reviewed_decisions`/`apply_reviewed_values` (in `Funding_dedup_helpers.py`) will raise an error rather than silently applying a partial result — this is a deliberate stale-file guard, not a bug.

### S5 — scope/pillar review notebook (1 review point)

| Review point | Exported file | Fill in | Save reviewed file as |
|---|---|---|---|
| Scope/pillar manual review | `data_review/<ts>_funding_for_review.csv` | `scope_curated` (`in`/`out`) and `pillar_curated` (PB/F/CM/CC, or blank/NA if out of scope) | `<ts>_funding_reviewed.csv` |

**Watch out:** the notebook cell that reads this back has a literal example path already typed into it from a previous real run, e.g.:
```python
reviewed_data = pd.read_csv("data_review/260729_2124_funding_reviewed.csv")
```
There's a comment above it telling you to update the date/timestamp — but it's easy to run the cell without editing it. If you do, it'll either error (if that old file no longer exists) or, worse, silently re-apply someone else's old reviewed file if one still happens to exist at that path. Always double-check that filename matches your own export before running that cell. Auto-curation logic for context: confidence 5–7 with a real pillar auto-approves `scope_curated='in'`; confidence 1 with no pillar auto-rejects `scope_curated='out'` (this means "excluded from promotion," not deleted — deleting would break S1's dedup-by-ID check on a future run); confidence 2–4 or anything inconsistent goes to manual review.

### S6 — LLM labelling script (1 review point)

| Review point | Exported file | Fill in | Save reviewed file as |
|---|---|---|---|
| Dimensions funder type | `data_review/<RUN_LABEL>_dimensions_funder_type_for_review.csv` | `Funder type` (Government, Nonprofit, For profit, etc.) | same name with `.csv` replaced by `_REVIEWED.csv` |

Dimensions carries no funder-sector signal at all (unlike Grants Tracker), so every newly-promoted Dimensions grant has a blank `Funder type` until someone fills this in. Every S6 run re-checks all past `*_for_review.csv`/`*_REVIEWED.csv` pairs sitting in `data_review/` and applies anything now resolved, before exporting a fresh snapshot of whatever's still blank — so you can leave this pending across multiple runs and it'll keep picking up your answers. Rows resolved to Government/Nonprofit also get `Total amount` copied into `Gov & NP contribution` automatically.

### S7 — data transformation notebook (1 optional review point)

| Review point | Exported file | Fill in | Save reviewed file as |
|---|---|---|---|
| LLM label completeness | `data_review/<ts>_llm_label_review.csv` | `reviewed_value` — leave blank to confirm the current value is correct, type a corrected label (matching the stage's delimiter: comma for research category, semicolon for end product/award purpose), or type `not required`/`skip` to leave alone, or `out of scope` to **permanently delete the row** (recorded in `manual_exclusions` so it can't resurface) | `<ts>_llm_label_review_REVIEWED.csv` |

This one is explicitly optional — the notebook says "safe to skip entirely."

---

## Part 9: LLM Labelling Stages (S6) in Detail

S6 runs up to 4 independent Batch API submissions per invocation, each gated by its own `RUN_*` flag, against two pools of rows: **Pool A** (newly scope-curated `funding_classified` rows not yet promoted) and **Pool B** (existing `funding_curated` rows missing this label, when `LABEL_SCOPE='new_only'`).

| Stage | Flag | Prompt(s) | Categories | Notes |
|---|---|---|---|---|
| Research category | `RUN_RESCAT` | `rescat_prompt_grants_{PB,F,CM,CC}.md` (per pillar) | 10–15 pillar-specific categories, e.g. PB: Crop development, Strain development, Ingredient optimisation, … | Multi-label (booleans); comma-joined; `Other` suppressed if any specific category is also true. |
| End product | `RUN_ENDPRODUCT` | `endproduct_prompt_grants.md` (shared, all pillars) | Meat, Fish and seafood, dairy subcategories (→ collapse to `Dairy`), Agnostic, Chocolate/desserts, Eggs, Spreads/sauces, … | Semicolon-joined; dairy specifics go in `sub-end product`; `Agnostic` suppressed if a specific category is also true. |
| Award purpose | `RUN_AWARDPURPOSE` | `awardpurpose_prompt_grants.md` (shared) | Research and development (default), Education and training, Networking, Equipment and infrastructure, Research infrastructure | Semicolon-joined. |
| Subpillar | `RUN_SUBPILLAR` | `subpillar_prompt_grants_{F,PB,CC}.md` | F: BF/PF (biomass vs. precision fermentation, multi-label); PB: Traditional fermentation (single boolean, exception not default); CC: Broad R&D / Technical research / Socioeconomic (single-choice) | **No subpillar prompt for Cultivated** — not an omission, cultivated grants simply have no subpillar concept. |

**Retry/refusal handling:** a row is retried (up to `MAX_RETRY_ATTEMPTS`, default 1) if it was never attempted, or its last attempt errored or hit `max_tokens`. `stop_reason='refusal'` is **never** auto-retried — it's treated as a genuine content refusal and routed straight to manual review (S7's label-completeness check) instead.

**Promotion:** any Pool A row with an attempted `rescat` result (success or failure — failures still get promoted with a blank value, so they're picked up for retry automatically on the next run) is promoted from `funding_classified` into `funding_curated`, with `Database='Dimensions'`, `Funding decision='Awarded'`, `AP pillar` derived from the pillar code, and PI/collaborator fields split out.

`funding_curated` is rewritten wholesale (`CREATE OR REPLACE`) at the end of every S6 run — see the troubleshooting note in Part 11 about what that implies.

---

## Part 10: Retrieving the Final Dataset

Run `S7_data_transformation.ipynb` (safe to re-run any time — it only fills gaps, never overwrites populated cells):

1. **Currency backfill, tier 1a** — re-queries Dimensions directly for any row missing a currency amount, in batches of 400 IDs.
2. **Currency backfill, tier 1b** — falls back to the Frankfurter FX API (free, ECB-backed, no key needed) using the historical rate on the grant's start date. **Known gotcha:** Frankfurter returns 403 on Python's default User-Agent — the notebook already spoofs a browser User-Agent to work around this, but if you ever see a wall of 403s here, that's the first thing to check.
3. **Per-year money split** — evenly splits `Gov & NP contribution (EUR)` across each year in `Years active` to populate the `2020`–`2035` per-year columns.
4. **QA checks** — flags LLM label completeness gaps (optional review, see Part 8) and any row with a partially-populated set of funding columns (a structural bug indicator, not a normal data gap).

Final export: `data_output/<timestamp>_funding_curated_FINAL.xlsx`.

Then run `Funding_analysis.ipynb` for reporting/analysis on top of the curated dataset.

---

## Part 11: Troubleshooting

See the combined manual's troubleshooting section first (conda activation, `ModuleNotFoundError`, missing API keys, "database file is being used by another process," restarting a run from scratch). Funding-specific additions:

- **Frankfurter API returns 403** — see Part 10, item 2. Fixed by the User-Agent spoof already in `S7_data_transformation.ipynb`; if it recurs, the API may have changed its blocking rules.
- **`funding_curated` looks like it lost data after a run** — both S2 and S6 rebuild this table from scratch (`CREATE OR REPLACE`) every time they run, from the raw last-report/Grants-Tracker files plus whatever's newly promoted. If a prior cycle's promoted grants aren't reflected in this cycle's raw input spreadsheets, a rebuild can silently drop them. S2 includes a read-only sanity check for exactly this (comparing previously-promoted `funding_classified` grants against the rebuilt `funding_curated`) — check its output if you're missing a grant you know was curated before.
- **A grant keeps reappearing after you excluded it** — check `manual_exclusions` (keyed by `join_key`); both S2 and S6 re-apply this table on every rebuild specifically so a deliberately-removed grant can't resurrect. If it's still coming back, confirm the row's `join_key` actually matches what's in the table.
- **`apply_reviewed_decisions`/`apply_reviewed_values` raises an error about a missing or stale match key** — your reviewed CSV's filename or contents don't match what the notebook expects (see Part 8's exact filename table per review point). This is a deliberate guard, not a bug — re-export and re-review rather than trying to patch around it.
- **A grant's label stays blank no matter how many times you re-run S6** — check `stop_reason_LLM` (or the per-stage equivalent). If it's `refusal`, it won't auto-retry; it needs a manual decision via S7's label-completeness review (Part 8).

---

## Part 12: Suggested Documentation Fixes (flagged, not applied)

Found while writing this manual — not changed, listed here for you to decide on:

1. **Inconsistent reviewed-filename casing across S2's 6 review points.** The funder-type review point expects `_REVIEWED.csv` (uppercase); all 5 title-match/pillar review points expect `_reviewed.csv` (lowercase). This isn't a functional bug (the read-back isn't case-sensitive matching logic — it's a literal path typed into each cell), but it's an easy mistake, especially across macOS/Linux where the filesystem *is* case-sensitive. Worth either standardizing the convention or at minimum adding an explicit comment above each export cell stating the exact expected suffix.
2. **`S5_Review_classifications.ipynb`'s read-back cell has a hardcoded example filename** (`260729_2124_funding_reviewed.csv`) left over from a real past run. There's a comment telling you to update it, but nothing stops you from running the cell without doing so. Consider either blanking it to something that errors loudly (e.g. `"REPLACE_ME.csv"`) or adding an assertion that checks the file's timestamp is recent.
3. **The Dimensions/`dimcli` login requirement isn't documented in the combined Publications/Patents manual.** `Funding/S1_query_dimensions.py`'s own comment notes that logging in requires a `dsl.ini` file on the machine in addition to the API key — this applies to all three pipelines (they all use `dimcli`), not just Funding, so it's arguably a gap in the *combined* manual rather than something to add here.
4. **Unresolved fuzzy-match discrepancy in S2.** The notebook's own diagnostic cell for the Dimensions↔last-report fuzzy title match found only 6 matches at threshold 85 in a recent run, versus 73 reportedly found by a prior manual pass. The notebook prints match counts at multiple thresholds/scorers to help diagnose this but doesn't resolve it — worth investigating before trusting that review point's completeness on a given run.
5. **`Funding/data_analysis/` is not in `.gitignore`**, unlike every other `data_*` folder under `Funding/`. Its two current files are meeting-report snapshots — but no current script actually writes to this folder (`export_combined_report.py`, the only likely candidate, writes to `data_audit/` instead). These are probably leftovers from an earlier version of that script pointing at `data_analysis` before it was changed. Worth deciding whether to delete the two stale files, or repoint the script back at `data_analysis` if that was the intended output location all along.
