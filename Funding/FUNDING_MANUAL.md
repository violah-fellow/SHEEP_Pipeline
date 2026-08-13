# Funding Pipeline Manual

For running the automated grants/funding data collection, deduplication, scoping, and labelling pipeline (`Funding/`).

This pipeline shares its API keys, conda environment, and general repo setup with the Publications and Patents pipelines — see the combined **Publications/Patents step-by-step manual** for all of that. This document only covers what's different about Funding, which is a lot: there is no ML classification step, every deduplicated grant goes straight to an LLM, and there are roughly **twelve separate manual review checkpoints** spread across three notebooks. If you've only ever run Publications or Patents, read this whole document before your first run — the shape of the pipeline is genuinely different, not just a renamed copy.

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

> **⚠️ The single biggest behavioral difference from Publications/Patents: `funding_curated` is rebuilt from scratch, not appended to.** Every time S2 or S6 runs, `funding_curated` is dropped and recreated (`CREATE OR REPLACE TABLE`) from the raw `raw_data/` spreadsheets plus whatever's newly promoted — it is not an append-only/upsert table the way `publications_classified`/`patents_classified` are. This is generally safe as long as your raw spreadsheets already reflect the previous cycle's curation, but it does mean the table's history isn't preserved run-to-run the way you might expect, and a manually-excluded grant can resurface in a later rebuild — S7 has a mandatory review that checks for exactly this every time it runs (Part 10.6, step 3b). **See Part 5 for exactly what triggers a rebuild and how it's guarded, and Part 11 if you ever see data that looks like it's disappeared.**

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
├── funding.db (+ .db.wal)           DuckDB database — self-contained to Funding/ (see Part 5)
├── llm_prompts/                     system prompts for every LLM stage (see Part 10)
├── raw_data/          [create by hand]  source spreadsheets (Grants Tracker, last report)
├── data_audit/         [auto-created]    before/after highlighted-diff Excel audit trails
├── data_review/        [auto-created]    manual-review CSV exports / reviewed files
├── data_output/        [auto-created]    final Excel exports (funding_curated, failures, etc.)
│   └── temp_data_analysis/    old, no-longer-regenerated meeting-report snapshots, moved here
│                               from a former top-level data_analysis/ folder (now gitignored
│                               along with the rest of data_output/*)
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

`Funding/data_output/temp_data_analysis/` holds a couple of old, no-longer-regenerated meeting-report snapshots — already covered by the `data_output/*` entry above, nothing extra needed.

---

## Part 5: Database Tables Reference

Everything lives in one DuckDB file, `funding.db`. These are the tables you'll actually encounter, roughly in the order they appear across a run:

| Table | Created by | Schema | Lifecycle |
|---|---|---|---|
| `<RUN_TABLE>` (e.g. `run_260811_2201`) | S1 | Dimensions DSL native field names (`id`, `title`, `abstract`, `funder_org_name`, `funding_usd`, …) | One per pipeline run. Created fresh (`CREATE OR REPLACE`) each time S1 runs; never touched again after S2 reads from it. |
| `<RUN_TABLE>_dedup` (`DEDUP_TABLE`) | S2 | Legacy "tracker" schema (`Grant ID`, `Title translated`, …) | One per run. Holds only the grants from this run's `<RUN_TABLE>` that survived all 7 dedup passes (genuinely new). This is what S3 reads. |
| `funding_classified` (`CLASSIFICATION_TABLE`) | S3 (created here if missing) | Starts as `DEDUP_TABLE`'s schema, then S3 adds `scope_LLM`/`confidence_LLM`/`pillar_LLM`/4 pillar booleans/`status_LLM`/`stop_reason_LLM`/`date_LLM`; S5 adds `scope_curated`/`pillar_curated`/`date_review`; S6 adds per-stage diagnostic columns (e.g. `research_category_LLM`, `category_status_LLM`, `category_retry_count_LLM`, …) | **Accumulates across every run** — this is the pipeline's permanent staging ledger for every Dimensions grant ever pulled, in scope or not. Rows are appended/updated, never deleted. |
| `funding_curated` (`CURATED_TABLE`) | S2 (first build) and S6 (every promotion) | The final GFI-tracker schema (~80 columns) — matches the shape of `Funding<year>_inscope.xlsx` | **Rebuilt from scratch (`CREATE OR REPLACE`) every time S2 or S6 runs** — see below, this is the single most important behavioral difference from Publications/Patents. |
| `funding_curated_pre_relabel_<RUN_LABEL>` | S6 | Snapshot of `funding_curated` immediately before a relabel | Only created when `LABEL_SCOPE='all'` on a fresh (non-resumed) S6 run — a full relabel overwrites `date_curated` on every row, so this snapshot is the only record of what the table looked like beforehand. Also exported to `data_output/<RUN_LABEL>_funding_curated_pre_relabel_snapshot.xlsx`. |
| `manual_exclusions` | S7 only (`CREATE TABLE IF NOT EXISTS`) | `join_key VARCHAR PRIMARY KEY`, `reason VARCHAR`, `date_added VARCHAR`, `title VARCHAR`, `lrd_row_id VARCHAR`, `identification_code VARCHAR`, plus (all `VARCHAR`) `funder_name`, `total_amount`, `total_amount_usd`, `total_amount_eur`, `gov_np_contribution`, `gov_np_contribution_usd`, `gov_np_contribution_eur`, `currency`, `project_start_date`, `project_start_year` | Permanent, human-readable audit log. Created within S7 during a final QA check of LLM labelling completeness, when a grant is marked "out of scope" — its row carries enough detail (title, both ID types, funder, every funding amount/currency, project dates) to judge whether a resurfaced match is genuinely the same grant. Read by the mandatory resurfacing review (Part 10.6, step 3b) every time S7 runs, which checks whether any previously excluded grant has reappeared in `funding_curated` and lets you remove it again if so. |

### `funding_curated`: rebuilt every run, not appended to

This is the one thing about Funding's data model with no equivalent in Publications or Patents, so it's worth spelling out precisely:

- **What triggers a rebuild:** every run of `S2_grant_deduplication.ipynb`, and every run of `S6_LLM_labelling.py` (including via `pipeline_funding.py`'s Step 5). Both execute `db.sql(f"CREATE OR REPLACE TABLE {CURATED_TABLE} AS SELECT * FROM ...")` — the existing table is dropped and replaced wholesale, not appended to or upserted into.
- **What it's rebuilt from, each time:** the two raw spreadsheets in `raw_data/` (last report + Grants Tracker), gap-filled and merged with whatever Dimensions data matched them, plus — in S6 — any newly-promoted Dimensions grants. A grant excluded via S7's label-completeness review can resurface in a later rebuild (e.g. if it's still sitting in the Grants Tracker source, or gets independently re-promoted). This is why S7 has a mandatory review (Part 10.6, step 3b) that checks every excluded grant against the current `funding_curated` on every run and lets you re-exclude it if it's come back.
- **Why this is usually safe:** the raw spreadsheets are expected to already reflect the previous cycle's curated state (`Funding<year>_inscope.xlsx` *is* last cycle's `funding_curated` export), so a rebuild should reproduce the same data plus whatever's genuinely new.
- **Where it can go wrong:** if the raw spreadsheets fed into a given run are stale — don't yet reflect a very recent promotion or hand-edit — the rebuild will silently reproduce the *old* state, because there's no append-based safety net underneath it. S2 includes a read-only sanity check specifically for this (comparing previously-promoted `funding_classified` grants against the freshly-rebuilt `funding_curated` — see Part 10.2, step 17) — always check its output.
- **Practical implication:** treat `funding_curated` as a derived/computed view of "raw spreadsheets + accumulated promotions," not as a table you can safely hand-edit in the database and expect to persist. A hand-edit will be silently overwritten on the next S2 or S6 run unless it's also reflected in the raw spreadsheets — and, for a manual exclusion specifically, unless S7's mandatory resurfacing review catches it and you confirm re-excluding it.

See Part 11 for what to do if you suspect a rebuild has dropped data you expected to see.

---

## Part 6: The Three Datasets and Their Column Maps

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

Whatever survives all 7 passes is genuinely new and becomes the `DEDUP_TABLE` (`<RUN_TABLE>_dedup`) that S3 picks up. See Part 10.2 for the full step-by-step narrative these passes sit within.

---

## Part 7: CONFIG Variables to Review Before Running

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

## Part 8: Running the Pipeline

Use the combined manual's `screen` (macOS) / WSL2 (Windows) guidance for keeping a long-running process alive. The difference for Funding: **expect to run the script three separate times per cycle**, not once:

```
conda activate sheep
cd Funding
python pipeline_funding.py
```

1. **First run** — executes Step 1 (query), then stops at Step 2 with printed instructions to run `S2_grant_deduplication.ipynb`.
2. Open and run `S2_grant_deduplication.ipynb` top to bottom (working through its 8 manual review points — see Part 9). Confirm it wrote a table named `<RUN_TABLE>_dedup` into `funding.db`.
3. **Second run** — `python pipeline_funding.py` again. It detects the dedup table exists, marks Step 2 done, runs Step 3 (LLM scope, may take a while as it polls the Batch API every 10 minutes), then stops at Step 4 with instructions to run `S5_Review_classifications.ipynb`.
4. Open and run `S5_Review_classifications.ipynb` (see Part 9). Confirm every grant from this run has a `scope_curated` decision.
5. **Third run** — `python pipeline_funding.py` again. It detects the review is complete, marks Step 4 done, and runs Step 5 (LLM labelling + promotion). When it finishes, you'll see `Pipeline complete for run '<RUN_TABLE>'.` followed by a boxed **"ACTION REQUIRED: run S7_data_transformation.ipynb now"** reminder — the pipeline being "complete" only means S1/S3/S6 have run; `funding_curated` isn't actually finished until S7's currency backfill and QA checks have been done. S7 isn't gated the way S2/S5 are (nothing downstream in this script depends on it, so there's no hard stop) — it's a reminder, not a checkpoint, so don't skip it just because the script didn't force you to.

**Resuming after an interruption:** just re-run `python pipeline_funding.py` — `RESUME=True` by default, so it finds the most recent incomplete run in `status_logs/` and continues from wherever it left off. **Restarting a run from scratch:** delete its `status_logs/status_<RUN_TABLE>.json` file (this does *not* clean up the corresponding DB tables — they're just orphaned).

Afterwards, run `S7_data_transformation.ipynb` by hand (currency backfill + QA — see Part 10.6) and, when you want the latest numbers, `Funding_analysis.ipynb`.

---

## Part 9: Manual Review Deep-Dive

This is the part that differs most from Publications/Patents — there are review checkpoints in three different notebooks. For every one of them, the pattern is the same: **the notebook cell that reads your reviewed file back in takes a literal filename you type in** — it is not a wildcard/glob lookup. So an exact filename match matters, and on a case-sensitive filesystem (macOS/Linux) getting the case wrong will throw a `FileNotFoundError`; on Windows (case-insensitive) it'll happen to work anyway, which can mask the mistake until someone else on a different OS hits it. Every review point now uses the same convention consistently — replace `_for_review.csv` with `_reviewed.csv` — but it's still worth a glance at the table below before saving, in case that ever drifts again. For the full step-by-step context each of these review points sits within, see Part 10.

### S2 — deduplication notebook (8 review points)

| Review point | Exported file | Fill in | Save reviewed file as | Gotchas |
|---|---|---|---|---|
| LRD missing date/year | `data_review/<ts>_lrd_missing_date_for_review.csv` | `corrected_year_project_started` and/or `corrected_project_start_date` (either, or both) | `<ts>_lrd_missing_date_reviewed.csv` | Only exported for rows with a nonzero value in any of the six funding columns — grants with no known funding stay silently excluded, exactly as before. Leaving both columns blank confirms exclusion; nothing rescues a row you don't fill in. Runs *before* the year-range filter, since a row with neither field has nothing to derive a year from and would otherwise be dropped before this review ever saw it. |
| GT missing date/year | `data_review/<ts>_gt_missing_date_for_review.csv` | `corrected_year_project_started` and/or `corrected_project_start_date` | `<ts>_gt_missing_date_reviewed.csv` | Same pattern as the LRD row above, but gated on the four native Grants Tracker funding columns (Gov & NP hasn't been derived yet at this point in the notebook) — there's no native EUR field on this side to check at all. |
| Grants Tracker funder type | `data_review/<ts>_gt_funder_type_for_review.csv` | `EXT_Funder type` (e.g. Government, Nonprofit, For profit) | `<ts>_gt_funder_type_reviewed.csv` | Leave blank if truly unknown. |
| GT ↔ last report, exact title match | `data_review/<ts>_gt_title_match_for_review.csv` | `is_true_match` (pre-filled `True`) — flip to `False` for any pair with a different start date, funding amount, Dimensions ID, or other distinguishing feature | `<ts>_gt_title_match_reviewed.csv` | |
| GT ↔ last report, fuzzy title match | `data_review/<ts>_gt_partial_title_match_for_review.csv` | same `is_true_match` pattern | `<ts>_gt_partial_title_match_reviewed.csv` | Confirmed matches just remove the duplicate — no gap-fill, too uncertain. |
| Dimensions ↔ last report, exact title match | `data_review/<ts>_dim_title_match_for_review.csv` | `is_true_match` | `<ts>_dim_title_match_reviewed.csv` | **Must save as UTF-8** — the notebook says so explicitly. Confirmed matches overwrite `Identification code` with the Dimensions ID (old value preserved to `Legacy identification code`). |
| Dimensions ↔ last report, fuzzy title match | `data_review/<ts>_dim_partial_title_match_for_review.csv` | `is_true_match` | `<ts>_dim_partial_title_match_reviewed.csv` | |
| AP pillar normalization | `data_review/<ts>_ap_pillar_for_review.csv` | `corrected_ap_pillar` — one of Plant-based / Fermentation / Cultivated / Cross-cutting | `<ts>_ap_pillar_reviewed.csv` | Only rows `normalize_ap_pillar()` couldn't auto-resolve show up here (casing/suffix variants and comma-joined multi-pillar values are handled automatically). |

If a reviewed file doesn't match the exact expected filename/casing, or is missing a row the notebook expects, `apply_reviewed_decisions`/`apply_reviewed_values` (in `Funding_dedup_helpers.py`) will raise an error rather than silently applying a partial result — this is a deliberate stale-file guard, not a bug.

### S5 — scope/pillar review notebook (1 review point)

| Review point | Exported file | Fill in | Save reviewed file as |
|---|---|---|---|
| Scope/pillar manual review | `data_review/<ts>_funding_for_review.csv` | `scope_curated` (`in`/`out`) and `pillar_curated` (PB/F/CM/CC, or blank/NA if out of scope) | `<ts>_funding_reviewed.csv` |

**Watch out:** the notebook cell that reads this back has a placeholder filename that will error loudly if you forget to change it:
```python
reviewed_data = pd.read_csv("data_review/REPLACE_ME.csv")
```
There's a comment next to it telling you to set this to match the reviewed file you saved — always double-check that filename matches your own export before running that cell. Auto-curation logic for context: confidence 5–7 with a real pillar auto-approves `scope_curated='in'`; confidence 1 with no pillar auto-rejects `scope_curated='out'` (this means "excluded from promotion," not deleted — deleting would break S1's dedup-by-ID check on a future run); confidence 2–4 or anything inconsistent goes to manual review.

### S6 — LLM labelling script (1 review point)

| Review point | Exported file | Fill in | Save reviewed file as |
|---|---|---|---|
| Dimensions funder type | `data_review/<RUN_LABEL>_dimensions_funder_type_for_review.csv` | `Funder type` (Government, Nonprofit, For profit, etc.) | `<RUN_LABEL>_dimensions_funder_type_reviewed.csv` |

Dimensions carries no funder-sector signal at all (unlike Grants Tracker), so every newly-promoted Dimensions grant has a blank `Funder type` until someone fills this in. Every S6 run re-checks all past `*_for_review.csv`/`*_reviewed.csv` pairs sitting in `data_review/` and applies anything now resolved, before exporting a fresh snapshot of whatever's still blank — so you can leave this pending across multiple runs and it'll keep picking up your answers. Rows resolved to Government/Nonprofit also get `Total amount` copied into `Gov & NP contribution` automatically.

### S7 — data transformation notebook (2 review points, both mandatory)

| Review point | Exported file | Fill in | Save reviewed file as |
|---|---|---|---|
| LLM label completeness | `data_review/<ts>_llm_label_for_review.csv` | `reviewed_value` — leave blank to confirm the current value is correct, type a corrected label (matching the stage's delimiter: comma for research category, semicolon for end product/award purpose), type `not required`/`skip` to explicitly record "no label needed" (functionally the same as blank), or `out of scope` to **delete the row now** and log it to `manual_exclusions` (see the resurfacing review below) | `<ts>_llm_label_reviewed.csv` |
| Manual-exclusion resurfacing | `data_review/<ts>_manual_exclusions_resurfaced_for_review.csv` | `still_out_of_scope` (pre-filled `True`) — leave `True` to confirm this excluded grant really has come back and should be removed again; flip to `False` for a false-positive match or if you've decided to let it back into scope (either way, the original `manual_exclusions` record is left untouched) | `<ts>_manual_exclusions_resurfaced_reviewed.csv` |

Both review points run every time S7 executes — neither is skippable. Because `funding_curated` is rebuilt from scratch on every S2/S6 run (see Part 5), a grant marked out of scope in the label-completeness review can still reappear if it's still sitting in the raw last-report/Grants Tracker source data; the resurfacing review is what catches that. It checks every grant ever logged to `manual_exclusions` against the current `funding_curated`, two independent ways: exact `Identification code` match, or exact normalized-title match (via `normalize_title`). A grant matching on both still produces exactly one row here, not a duplicate — the exported CSV's `matched_via` column lists every criterion that hit. To judge whether a flagged match is genuinely the same grant, the export pairs an `excluded_*` column (what was recorded when it was originally excluded) against a `current_*` column (what's in `funding_curated` right now) for every comparison field — title, funder name, all 6 funding-amount columns, currency, project start date/year — so nothing needs cross-referencing by hand.

---

## Part 10: Script-by-Script Walkthrough

This section walks through what every S1–S7 file actually does, end to end. It's most detailed for S2 — by far the most complex file in the pipeline — and briefer for the others. For the exact manual-review filenames/gotchas referenced below, see Part 9.

### 10.1 — S1: Query Dimensions (`S1_query_dimensions.py`)

Purely automated, no manual review, no LLM calls.

1. Loads `DIMENSIONS_API_KEY` from `.env`, logs into `dimcli`.
2. Reads search strings from `dimensions_search_funding.txt` (one per line).
3. For each search string, runs `dsl.query_iterative()` against Dimensions' `grants` full-text search (`title_abstract_only`), filtered to `start_year` within `[START_YEAR:END_YEAR]`. Pulls ~28 fields per grant: `id`, `title`, `original_title`, `abstract`, `start_date`/`start_year`, `end_date`, funder fields (`funder_orgs`, `funder_org_name`, `funder_org_countries`, `funder_org_cities`), funding amounts in 8 currencies (`funding_usd`, `funding_eur`, `funding_gbp`, `funding_aud`, `funding_cad`, `funding_chf`, `funding_jpy`, `funding_nzd`) plus `funding_currency` (metadata only — Dimensions deprecated a generic native-amount field in 2018, hence pulling all 8), research org/researcher fields, `keywords`, `linkout`, `dimensions_url`, `category_for_2020`, `category_sdg`.
4. Concatenates all search-string results into one dataframe, dedupes by `id`, stamps `date_dimensions` = today (YYMMDD).
5. Filters out any grant `id` already present in `funding_classified`'s `"Grant ID"` column — the pipeline's first, cheapest dedup pass (comparing DSL's `id` directly against the legacy tracker's `"Grant ID"` column; same `grant.NNNNNNNN` values, different column names either side).
6. Writes survivors to a fresh table `<RUN_TABLE>` (`CREATE OR REPLACE TABLE`).

### 10.2 — S2: Deduplication & 3-Way Merge (`S2_grant_deduplication.ipynb`)

The most complex file in the pipeline. Manual notebook — run top to bottom.

The config cell sets `RUN_TABLE`/`RUN_TIMESTAMP` (must match S1's output), `LAST_REPORT_FILE`/`GRANTS_TRACKER_FILE` (the two raw spreadsheets), `LAST_REPORT_SCOPE` (see note below), `FUZZY_THRESHOLD` (85), and `FUNDING_START_YEAR_MIN`/`MAX` (2020/2025).

> **Note on `LAST_REPORT_SCOPE`:** default should now be `'global'`. This was designed as a one-off transition flag, not a permanent toggle. Setting it to `'europe'` let the pipeline import newly-added/modified European Grants Tracker entries into the last-report dataset while *also* pulling in every global (non-Europe) Grants Tracker grant within the project-start-year filter, regardless of when it was last modified — useful at the time because the last report itself was Europe-only, so there was no historical global data to build on yet. That one-off global backfill has now happened: all of that historical global data is already in the data output, so from here on every run only needs to filter for *recently modified* Grants Tracker entries, Europe or otherwise — hence `'global'` (step 2 below refers back to this rather than repeating it).

1. **Load last report** (`raw_data/Funding<year>_inscope.xlsx`). Renames a couple of legacy columns (`Production platform`→`AP pillar`, etc.), stamps a surrogate ID (`lrd_row_id`), then derives `Year project started` from `Project start date` wherever blank (via `extract_year`). Rows still missing both the date and year-started fields, but with a real funding amount, are exported for manual review (see Part 9). After that, the project-start-year filter is applied (2020–2025 at the time of writing) — anything still blank, or outside this range, is excluded (no date at all just isn't worth keeping).
2. **Load Grants Tracker** (`raw_data/GrantsTracker_*.xlsx`). Stamps a surrogate ID (`gt_row_id`), then derives `EXT_Year project starts` from `EXT_Project start date (estimated)` wherever blank, same as step 1. Rows still missing both fields, but with a real funding amount, are exported for manual review (see Part 9). After that, two filters are combined and applied together: a date-window filter on `EXT_Last modified` (falling back to `EXT_Date added` where blank) — the assumption being the previous run already collected everything up to that point, so only what's changed since needs re-checking, avoiding unnecessary dedup effort — and the same project-start-year filter as step 1. See the `LAST_REPORT_SCOPE` note above for the one exception to the date-window filter (a one-off Europe/global carve-out, now retired).
3. **Country cleaning** — `EXT_PI organization country` is deduped/alias-corrected and invalid entries dropped; `EXT_Funder country` is alias-normalized only (multi-value/`EU`/`Global` preserved).
4. **Adapt Dimensions data** — converts S1's DSL-native columns (`id`, `title`, `investigators`, `funding_usd`, …) into the legacy tracker schema the rest of the notebook expects (no-op if already in that schema).
5. **Dimensions ↔ last report, exact-ID gap-fill** — real business-key match on `Grant ID`==`Identification code`, no review needed. Exports a highlighted-diff audit to `data_audit/`.
6. **Dimensions ↔ Grants Tracker, exact-ID gap-fill** — same idea, different column map. Audit exported.
7. **Derive Grants Tracker's own Gov & NP contribution** — uses the real figure if nonzero; else infers from `Funder type` if Government/Nonprofit; else → **manual review point 1** (funder type, see Part 9).
8. **Grants Tracker ↔ last report, exact Dimensions-ID match** — trusted only for IDs matching `grant.NNNNNNNN`. No review needed.
9. **Manual review point 2** — Grants Tracker ↔ last report, exact normalized title match. Confirmed matches get the usual `gap_fill` (via `gt_lrd_col_map`) — no ID-overwrite logic here (that's specific to Dimensions matches, steps 12–13).
10. **Manual review point 3** — Grants Tracker ↔ last report, fuzzy title match (rapidfuzz ≥ 85). Confirmed matches just remove the duplicate — no gap-fill, considered too uncertain.
11. **Append leftover Grants Tracker rows** — anything that matched nothing gets appended into the last-report dataset as genuinely new rows (`Database='airtable'`), with a synthesized `lrd_row_id` (`'GT_' + gt_row_id`) to avoid key collisions downstream.
12. **Manual review point 4** — Dimensions ↔ last report, exact title match. Confirmed matches get the usual `gap_fill` (via `col_map`, filling any empty columns on the last-report row from the matched Dimensions row) **and** have `Identification code` overwritten with the Dimensions ID (old value preserved to `Legacy identification code` first, unless something's already stashed there) — this is the only ID format later automated matching trusts, so a non-Dimensions code sitting there just means the same grant needs re-matching by title every future run.
13. **Manual review point 5** — Dimensions ↔ last report, fuzzy title match (`FUZZY_THRESHOLD = 85`, no multi-threshold comparison — a straight cutoff). No general column gap-fill here (considered too uncertain), but confirmed matches still get `Identification code` set to the Dimensions ID — same "trusted ID format" reasoning as step 12, just without the broader gap-fill: filled if currently empty, or the old value preserved to `Legacy identification code` first if it differs.
14. **Manual review point 6** — AP pillar normalization, for anything `normalize_ap_pillar()` couldn't auto-resolve (casing/suffix variants and comma-joined multi-pillar values are handled automatically; only genuinely ambiguous values reach this review).
15. **Region derivation** — `Funder region`/`PI organisation region` are fully re-derived (not gap-filled) from country columns.
16. **Final writes:**
    - `<RUN_TABLE>_dedup` — genuinely-new Dimensions grants only, this is S3's input.
    - `funding_curated` — rebuilt from scratch (see Part 5) from the merged/gap-filled dataset.
    - An Excel mirror to `data_audit/`.
17. **Read-only sanity check** (no action needed unless it flags something): every grant already promoted to `funding_classified` should still appear in the rebuilt `funding_curated` — flags a silently-dropped promotion (this is the check referenced in Part 5's rebuild explanation). A second check that used to live here (Gov & NP contribution EUR completeness) has been removed — it operated on the whole table, so it was really validating S7's currency backfill rather than anything S2 itself does; S7 already has a more thorough equivalent (see Part 10.6) that checks all three currency variants for consistency, not just EUR in isolation.

### 10.3 — S3: LLM Scope Classification (`S3_LLM_scope.py`)

Automated, uses the Anthropic Batch API. No manual review in this script — that happens next, in S5.

1. Authenticates via `CLAUDE_API_KEY`.
2. Seeds/grows `funding_classified` from `DEDUP_TABLE` (idempotent — skips grants already present).
3. Builds one batch request per grant: system prompt = `scope_prompt_funding.md` (cached), user message = grant's title + abstract. Tool-forced to a `classify_grant` schema: `scope` (`in`/`out`), `confidence` (1–7), and 4 pillar booleans (`plant_based`, `fermentation`, `cultivated`, `cross_cutting`). Chunks requests into batches of 2000, saves submission metadata to `batch_jobs/<RUN_TABLE>_llm_scope.json` (so a resumed run doesn't resubmit).
4. Polls every 10 minutes (`POLL_INTERVAL_SECONDS=600`) until the batch reports `"ended"`.
5. Parses results — any incomplete/malformed tool call is coerced to `None` with `status_LLM='incomplete_tool_call'` rather than crashing.
6. Derives `pillar_LLM` from the 4 booleans: `CC` if more than one is true or only `cross_cutting` is true; otherwise whichever single one is true; otherwise `NA`.
7. Writes `scope_LLM`/`confidence_LLM`/`pillar_LLM`/the 4 booleans/`status_LLM`/`stop_reason_LLM`/`date_LLM` onto both `DEDUP_TABLE` and `funding_classified`.

### 10.4 — S5: Scope/Pillar Review (`S5_Review_classifications.ipynb`)

Manual notebook, run independently of `pipeline_funding.py`'s own step-4 gate whenever there's new LLM-scored data. No LLM calls here — purely human-decision application. Two sections:

**Section 1 — auto-curate:**
1. Loads `funding_classified` rows where `status_LLM` is set and `scope_curated` is still blank (so re-runs don't re-export already-curated rows).
2. Applies the confidence-band rule: confidence 5–7 with a real pillar → auto-approve `scope_curated='in'`; confidence 1 with no pillar → auto-reject `scope_curated='out'` (meaning "excluded from promotion," not deleted — deleting would break S1's dedup-by-ID check on a future run); anything else (confidence 2–4, or inconsistent) → flagged `'manual_review'`.
3. Exports the `'manual_review'` rows to `data_review/<ts>_funding_for_review.csv` (see Part 9 for the review itself).
4. Writes the auto-curated rows straight back into `funding_classified`.

**Section 2 — apply manual decisions (the manual step):**
5. You fill in the exported CSV and save it back (see Part 9's exact naming/gotchas).
6. The read-back cell uses `keep_default_na=False, na_values=['']` deliberately — pandas' default NA list includes the literal string `'NA'`, which would otherwise corrupt a legitimate `pillar_curated='NA'` (out-of-scope) value on load.
7. Applies your decisions back via `UPDATE ... FROM ... WHERE "Grant ID" = ...`.

A final validation/auto-fix cell (safe to re-run any time) strips stray whitespace, backfills `pillar_curated='NA'` for any out-of-scope row missing it, and flags — but doesn't fix — anything still invalid (bad `scope_curated` value, an in-scope row with an unrecognised pillar, an out-of-scope row with a pillar other than `'NA'`, or a row still stuck at `'manual_review'`).

### 10.5 — S6: LLM Labelling & Promotion (`S6_LLM_labelling.py`)

Automated, batch API, run via `pipeline_funding.py`'s Step 5. Runs up to 4 independent Batch API submissions per invocation, each gated by its own `RUN_*` flag, against two pools of rows: **Pool A** (newly scope-curated `funding_classified` rows not yet promoted) and **Pool B** (existing `funding_curated` rows missing this label, when `LABEL_SCOPE='new_only'`).

| Stage | Flag | Prompt(s) | Categories | Notes |
|---|---|---|---|---|
| Research category | `RUN_RESCAT` | `rescat_prompt_grants_{PB,F,CM,CC}.md` (per pillar) | 10–15 pillar-specific categories, e.g. PB: Crop development, Strain development, Ingredient optimisation, … | Multi-label (booleans); comma-joined; `Other` suppressed if any specific category is also true. |
| End product | `RUN_ENDPRODUCT` | `endproduct_prompt_grants.md` (shared, all pillars) | Meat, Fish and seafood, dairy subcategories (→ collapse to `Dairy`), Agnostic, Chocolate/desserts, Eggs, Spreads/sauces, … | Semicolon-joined; dairy specifics go in `sub-end product`; `Agnostic` suppressed if a specific category is also true. |
| Award purpose | `RUN_AWARDPURPOSE` | `awardpurpose_prompt_grants.md` (shared) | Research and development (default), Education and training, Networking, Equipment and infrastructure, Research infrastructure | Semicolon-joined. |
| Subpillar | `RUN_SUBPILLAR` | `subpillar_prompt_grants_{F,PB,CC}.md` | F: BF/PF (biomass vs. precision fermentation, multi-label); PB: Traditional fermentation (single boolean, exception not default); CC: Broad R&D / Technical research / Socioeconomic (single-choice) | **No subpillar prompt for Cultivated** — not an omission, cultivated grants simply have no subpillar concept. |

**Retry/refusal handling:** a row is retried (up to `MAX_RETRY_ATTEMPTS`, default 1) if it was never attempted, or its last attempt errored or hit `max_tokens`. `stop_reason='refusal'` is **never** auto-retried — it's treated as a genuine content refusal and routed straight to manual review (S7's label-completeness check) instead.

**Promotion:** any Pool A row with an attempted `rescat` result (success or failure — failures still get promoted with a blank value, so they're picked up for retry automatically on the next run) is promoted from `funding_classified` into `funding_curated`, with `Database='Dimensions'`, `Funding decision='Awarded'`, `AP pillar` derived from the pillar code, and PI/collaborator fields split out.

**How does a future run know what's already been labelled?** In plain terms: picture every grant as having 4 separate checkboxes, one per stage (research category, end product, award purpose, subpillar) — not a single "done/not done" switch for the whole grant. Each checkbox has its own tracking column, so the 4 stages are checked completely independently. On a `new_only` run, for each stage S6 looks at that grant's checkbox for that stage only: blank, or ticked-but-failed-with-retries-left → try again; ticked-and-succeeded (or ticked-and-permanently-failed) → leave it alone. So it's entirely normal for a grant to have research category done but subpillar still blank — it'll just keep showing up for subpillar on future runs until that one's ticked too.

While a grant hasn't been promoted yet, all 4 of its checkboxes live on `funding_classified`. The moment it *is* promoted (which, as above, only needs the research category checkbox to have been attempted), all 4 checkboxes move house to `funding_curated` — including any that were still blank at the moment of promotion. So a newly-promoted grant might land in `funding_curated` with, say, 3 of its 4 labels already filled in (carried over from `funding_classified` at promotion time), and the 4th gets picked up and filled in later by checking `funding_curated` directly, not by going back to `funding_classified` — that table stops being checked for this grant entirely once it's promoted.

`funding_curated` is rewritten wholesale (`CREATE OR REPLACE`) at the end of every S6 run — see Part 5 for the full mechanics of what that means.

### 10.6 — S7: Currency Backfill & QA (`S7_data_transformation.ipynb`)

Manual/standalone — not run by `pipeline_funding.py`, but the pipeline's own completion message reminds you to run it (see Part 8). Safe to re-run any time — it only fills gaps, never overwrites populated cells.

- **Step 1a — currency backfill, tier 1** — re-queries Dimensions directly for any row missing a currency amount, in batches of 400 IDs.
- **Step 1b — currency backfill, tier 2** — falls back to the Frankfurter FX API (free, ECB-backed, no key needed) using the historical rate on the grant's start date. **Known gotcha:** Frankfurter returns 403 on Python's default User-Agent — the notebook already spoofs a browser User-Agent to work around this, but if you ever see a wall of 403s here, that's the first thing to check.
- **Step 2 — per-year money split** — evenly splits `Gov & NP contribution (EUR)` across each year in `Years active` to populate the `2020`–`2035` per-year columns. Exports a snapshot to `data_output/<timestamp>_funding_curated_pre-review-snapshot.xlsx` — deliberately **not** called `_FINAL` yet, since it hasn't been through the two review steps below.
- **Step 3a — QA: LLM-label completeness check + manual review** (mandatory, see Part 9) — marking a row `out of scope` here deletes it from `funding_curated` immediately and logs it to `manual_exclusions`.
- **Step 3b — manual-exclusion resurfacing review** (mandatory, see Part 9) — every time this notebook runs, checks every grant ever logged to `manual_exclusions` (this run or any earlier one) against the current `funding_curated`, two independent ways: exact `Identification code` match or exact normalized-title match. Any hit is exported with side-by-side `excluded_*`/`current_*` comparison columns (title, funder, amounts, currency, dates) and a `still_out_of_scope` column (pre-filled `True`) for you to confirm or reject; confirmed rows are removed from `funding_curated` again. Exports the genuine final snapshot once this step completes: `data_output/<timestamp>_funding_curated_FINAL.xlsx`.
- **Step 4 — QA: funding-column completeness check** — flags a row where *some* (not all) of the 6 core funding columns are populated after the backfill in steps 1a–1b — a structural bug indicator, not a normal data gap.

Then run `Funding_analysis.ipynb` for reporting/analysis on top of the curated dataset.

---

## Part 11: Troubleshooting

See the combined manual's troubleshooting section first (conda activation, `ModuleNotFoundError`, missing API keys, "database file is being used by another process," restarting a run from scratch). Funding-specific additions:

- **Frankfurter API returns 403** — see Part 10.6, step 1b. Fixed by the User-Agent spoof already in `S7_data_transformation.ipynb`; if it recurs, the API may have changed its blocking rules.
- **`funding_curated` looks like it lost data after a run** — both S2 and S6 rebuild this table from scratch (`CREATE OR REPLACE`) every time they run, from the raw last-report/Grants-Tracker files plus whatever's newly promoted (full mechanics in Part 5). If a prior cycle's promoted grants aren't reflected in this cycle's raw input spreadsheets, a rebuild can silently drop them. S2 includes a read-only sanity check for exactly this (comparing previously-promoted `funding_classified` grants against the rebuilt `funding_curated` — Part 10.2, step 17) — check its output if you're missing a grant you know was curated before.
- **Gov & NP / currency-completeness gaps right after S2** — expected, not a bug: S2 doesn't check EUR completeness at all; that's S7's job, after its currency backfill has actually run (Part 10.6, steps 1–2). Wait until `S7_data_transformation.ipynb` has run before treating a currency gap as a real problem.
- **A grant keeps reappearing after you excluded it** — check whether S7's mandatory resurfacing review (Part 10.6, step 3b) flagged it. If it did and you (or someone else) flipped `still_out_of_scope` to `False`, that's why it's back — re-run the review and leave it `True` this time. If it *wasn't* flagged at all, the grant's `Identification code` may have changed and its title doesn't match exactly either (e.g. reworded in the source data); check `manual_exclusions` manually against the grant's current row, and re-exclude it via the LLM-label review's `out of scope` marker if confirmed.
- **`apply_reviewed_decisions`/`apply_reviewed_values` raises an error about a missing or stale match key** — your reviewed CSV's filename or contents don't match what the notebook expects (see Part 9's exact filename table per review point). This is a deliberate guard, not a bug — re-export and re-review rather than trying to patch around it.
- **A grant's label stays blank no matter how many times you re-run S6** — check `stop_reason_LLM` (or the per-stage equivalent). If it's `refusal`, it won't auto-retry; it needs a manual decision via S7's label-completeness review (Part 9).

This section stays as a place for future findings.
