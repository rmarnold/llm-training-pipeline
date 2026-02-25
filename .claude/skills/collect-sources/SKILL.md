---
name: collect-sources
description: "Phase 1: Collect and catalog training data sources for a language. Searches web for knowledge content, catalogs repos, generates safety scenario seeds."
disable-model-invocation: true
argument-hint: "[language] [category?]"
---

# /collect-sources

Phase 1 of the SOTA data pipeline. Uses Phase 0 configs to **find and catalog** training data sources for a target language. Produces a `source_manifest.jsonl` with full provenance metadata, fetched web content as markdown files, and safety scenario seeds.

Claude IS the oracle in this pipeline — it generates search queries, evaluates sources, and catalogs everything within the session.

## Strategy Dispatch

Each taxonomy capability has a `data_generation_strategy`. Handle each differently:

| Strategy | Category | What to Do | Tools |
|----------|----------|------------|-------|
| `web_extraction` | knowledge | Search web for docs/tutorials, fetch content, check license, detect PII, save markdown | WebSearch, WebFetch, Write |
| `sota_trajectory` | skills + procedural | Read `data_sources_{LANGUAGE}.yaml` repo list, verify licenses via `gh api`, map repos to capabilities | Read, Bash(gh) |
| `mutation_plus_sota` | skills (debugging) + procedural | Filter trajectory repos to those with test suites, mark `mutation_eligible: true` | Read, Bash(gh) |
| `sota_preference` | behavioral | Load safety scenario seeds, generate language-specific scenarios, search for safety policy docs | Read, WebSearch, Write |

## Instructions

### Step 1: Parse Arguments

Parse `$ARGUMENTS` for:
- `$ARGUMENTS[0]` -> `LANGUAGE` (required). Valid values: `rust`, `python`, `typescript`, `go`.
- `$ARGUMENTS[1]` -> `CATEGORY` (optional). Valid values: `knowledge`, `skills`, `procedural`, `behavioral`. If omitted, process ALL categories.

If `LANGUAGE` is missing or invalid, ask the user using AskUserQuestion with the 4 language options.

Normalize to lowercase and trim whitespace.

### Step 2: Read Configs

Read these files — they are required inputs created by Phase 0 (`/init-data-pipeline`):

1. `configs/sota/{LANGUAGE}_taxonomy.yaml` — capability list with `data_generation_strategy` per capability
2. `configs/sota/governance.yaml` — `license_policy` (permissive/attribution_required/blocked), `pii_scrubbing.patterns`, `contamination_checking.eval_benchmarks`
3. `configs/sota/provenance_schema.yaml` — required metadata fields for manifest entries
4. `configs/data_sources_{LANGUAGE}.yaml` — curated repo list with categories (if it exists; skip repo-dependent steps if not)
5. `.claude/skills/collect-sources/web_search_templates.md` — query templates per language per capability

If taxonomy or governance configs are missing, tell the user to run `/init-data-pipeline {LANGUAGE}` first and stop.

### Step 3: Initialize Manifest

Check if `data/sota/{LANGUAGE}/raw/source_manifest.jsonl` exists:
- **If yes**: Read all lines and extract `source_url` values into a set for URL dedup. Print "Loaded {N} existing manifest entries for dedup."
- **If no**: Create the file (empty) and the `data/sota/{LANGUAGE}/raw/web/` directory. Print "Initialized empty manifest."

Generate a `batch_id` using a UUID4. Record `started_at` as the current ISO-8601 timestamp.

Initialize counters: `total_added = 0`, `pii_flagged = 0`, `license_unknown = 0`, per-strategy counts.

### Step 4: Process `web_extraction` Capabilities (knowledge category)

**Skip this step** if `CATEGORY` is set and is NOT `knowledge`.

For each capability in the taxonomy where `data_generation_strategy == "web_extraction"`:

1. **Generate queries**: Use the templates in `web_search_templates.md` for the current `LANGUAGE` and capability. Adapt the template into 5-8 concrete search queries. Include the language name in each query.

2. **Search**: Run WebSearch for each query.

3. **Evaluate results**: For the top results (up to 10 unique URLs per capability):
   - Check URL not already in the manifest dedup set
   - Skip if URL is already known

4. **Fetch**: WebFetch each new URL. Extract the main content.

5. **License determination**:
   - Check `web_search_templates.md` "Known Permissive Docs Sites" section — if the URL's domain is listed, use that license.
   - For GitHub repos, extract `license.spdx_id` from the URL pattern.
   - For unknown sites, set `source_license: "unknown_needs_review"` and increment `license_unknown`.

6. **Contamination check**: Scan the fetched content for exact matches of any benchmark name from `governance.yaml → contamination_checking.eval_benchmarks` (e.g., "HumanEval", "MBPP", "SWE-bench"). Set `contamination_checked: true`. If a benchmark name appears in the content, add a note in the manifest entry but do NOT reject — Phase 2 handles deeper decontamination.

7. **PII detection**: Scan content against each regex pattern in `governance.yaml → pii_scrubbing.patterns`. If any pattern matches, set `pii_detected: true` and increment `pii_flagged`. Do NOT scrub — Phase 2 handles scrubbing. Set `pii_scrubbed: false`.

8. **Save content**: Write the fetched markdown to `data/sota/{LANGUAGE}/raw/web/{capability}_{index:03d}.md` where `index` is a zero-padded counter per capability (starting from the highest existing index + 1).

9. **Append manifest entry**: Write one JSONL line to `source_manifest.jsonl` with this schema:

```json
{
  "example_id": "<uuid4>",
  "pipeline_version": "1.0.0",
  "phase": "phase1_collection",
  "taxonomy_capability": "<capability_name>",
  "taxonomy_category": "knowledge",
  "language": "<LANGUAGE>",
  "source_url": "<fetched_url>",
  "source_license": "<determined_license>",
  "source_repo": null,
  "source_commit": null,
  "oracle_model": null,
  "oracle_temperature": null,
  "oracle_system_prompt_hash": null,
  "pii_scrubbed": false,
  "pii_detected": <true|false>,
  "contamination_checked": true,
  "dedup_hash": null,
  "data_generation_strategy": "web_extraction",
  "raw_file": "data/sota/<LANGUAGE>/raw/web/<capability>_<index>.md",
  "content_length_chars": <int>,
  "fetch_timestamp": "<iso8601>",
  "batch_id": "<batch_id>"
}
```

10. Add the URL to the dedup set. Increment `total_added` and the `web_extraction` counter.

Print progress after each capability: `"[web_extraction] {capability}: fetched {N} sources"`.

### Step 5: Process `sota_trajectory` Capabilities (skills + procedural categories)

**Skip this step** if `CATEGORY` is set and is NOT `skills` AND NOT `procedural`.

This step catalogs repos from `data_sources_{LANGUAGE}.yaml` and maps them to taxonomy capabilities.

1. **Read repo list**: Load `configs/data_sources_{LANGUAGE}.yaml`. Extract the `{LANGUAGE}_repos.categories` section (e.g., `rust_repos.categories`). For languages without a `*_repos` section, skip this step and print a note.

2. **Build repo-to-capability mapping**: Map repo categories to taxonomy capabilities:

   **Rust mappings** (adapt similarly for other languages):
   | Repo Category | Taxonomy Capabilities |
   |---------------|----------------------|
   | data_structures | code_generation, test_writing |
   | error_handling | debugging, code_generation |
   | utilities | code_generation, test_writing |
   | parsing | code_generation, planning |
   | string_handling | code_generation |
   | serialization | code_generation, planning |
   | testing | test_writing, debugging |
   | async_core | code_generation, state_tracking |
   | web | code_generation, tool_use, planning |
   | concurrency | debugging, state_tracking |

   For non-Rust languages, use equivalent mappings based on the repo categories present.

3. **Verify licenses**: For each unique repo, run:
   ```bash
   gh api repos/{owner}/{repo} --jq '.license.spdx_id // "NONE"'
   ```
   - Normalize the repo path: if the entry is a string like `"BurntSushi/bstr"`, use it directly. If it's an object with `repo:` key, use that.
   - Check the returned SPDX ID against `governance.yaml → license_policy`:
     - If in `permissive` or `attribution_required` lists: accept, record the SPDX ID
     - If in `blocked` list: reject, print warning, skip this repo
     - If `"NONE"` or not in any list: set `source_license: "unknown_needs_review"`, increment `license_unknown`
   - Rate limit: add a brief pause between `gh api` calls to avoid GitHub rate limiting.

4. **Get latest commit**: For each accepted repo:
   ```bash
   gh api repos/{owner}/{repo}/commits --jq '.[0].sha' -q
   ```

5. **Append manifest entries**: For each repo, create one entry per mapped capability:

```json
{
  "example_id": "<uuid4>",
  "pipeline_version": "1.0.0",
  "phase": "phase1_collection",
  "taxonomy_capability": "<mapped_capability>",
  "taxonomy_category": "<category_from_taxonomy>",
  "language": "<LANGUAGE>",
  "source_url": "https://github.com/<owner>/<repo>",
  "source_license": "<verified_spdx>",
  "source_repo": "<owner>/<repo>",
  "source_commit": "<latest_sha>",
  "oracle_model": null,
  "oracle_temperature": null,
  "oracle_system_prompt_hash": null,
  "pii_scrubbed": false,
  "pii_detected": false,
  "contamination_checked": false,
  "dedup_hash": null,
  "data_generation_strategy": "sota_trajectory",
  "repo_category": "<category_from_data_sources>",
  "package": "<package_name_if_workspace_repo_else_null>",
  "fetch_timestamp": "<iso8601>",
  "batch_id": "<batch_id>"
}
```

6. Increment `total_added` and the `sota_trajectory` counter per entry.

Print progress: `"[sota_trajectory] Cataloged {N} repo-capability pairs across {M} repos"`.

### Step 6: Process `mutation_plus_sota` Capabilities (debugging + multi_step_debugging)

**Skip this step** if `CATEGORY` is set and is NOT `skills` AND NOT `procedural`.

1. **Filter from Step 5**: From the repos cataloged in Step 5, identify those that are mutation-eligible:
   - The `data_sources_{LANGUAGE}.yaml` quality criteria (e.g., `cargo_test_passes: true` for Rust) already imply test suites exist.
   - For Rust, all repos in `rust_repos.categories` meet `cargo_test_passes: true` per the quality criteria.
   - For other languages, check the equivalent quality criterion in the data_sources file.

2. **Mark mutation-eligible**: For each mutation-eligible repo, append manifest entries for the `mutation_plus_sota` capabilities (`debugging`, `multi_step_debugging`):

```json
{
  "example_id": "<uuid4>",
  "pipeline_version": "1.0.0",
  "phase": "phase1_collection",
  "taxonomy_capability": "debugging",
  "taxonomy_category": "skills",
  "language": "<LANGUAGE>",
  "source_url": "https://github.com/<owner>/<repo>",
  "source_license": "<verified_spdx_from_step5>",
  "source_repo": "<owner>/<repo>",
  "source_commit": "<latest_sha_from_step5>",
  "oracle_model": null,
  "oracle_temperature": null,
  "oracle_system_prompt_hash": null,
  "pii_scrubbed": false,
  "pii_detected": false,
  "contamination_checked": false,
  "dedup_hash": null,
  "data_generation_strategy": "mutation_plus_sota",
  "mutation_eligible": true,
  "repo_category": "<category_from_data_sources>",
  "package": "<package_name_if_workspace_repo_else_null>",
  "fetch_timestamp": "<iso8601>",
  "batch_id": "<batch_id>"
}
```

   Create one entry per capability (`debugging` in skills category + `multi_step_debugging` in procedural category) per eligible repo.

3. **Check count**: If fewer than 10 repos are mutation-eligible, search for additional repos:
   - WebSearch for `"{LANGUAGE} open source repos with good test coverage site:github.com"` (adapt per language)
   - For each found repo, verify license with `gh api`, check test suite exists
   - Add to manifest if eligible

4. Increment `total_added` and the `mutation_plus_sota` counter.

Print progress: `"[mutation_plus_sota] Marked {N} repos as mutation-eligible across {M} capabilities"`.

### Step 7: Process `sota_preference` Capabilities (behavioral category)

**Skip this step** if `CATEGORY` is set and is NOT `behavioral`.

1. **Load seeds**: Read `.claude/skills/collect-sources/safety_scenario_seeds.md`.

2. **Generate scenarios**: For each seed category in the file:
   - Read the seed descriptions
   - Generate 20-30 **language-specific** scenario descriptions. Each scenario should be:
     - Concrete (references the LANGUAGE, its tools, common patterns)
     - Classified with `expected_behavior`: `refuse`, `clarify`, or `hedge`
     - Assigned a `difficulty`: `easy`, `medium`, or `hard`
   - Write scenarios to `data/sota/{LANGUAGE}/raw/safety_scenarios_{seed_category}.jsonl`
   - Each line:
     ```json
     {
       "scenario_id": "<uuid4>",
       "category": "<seed_category>",
       "description": "<concrete scenario>",
       "expected_behavior": "refuse|clarify|hedge",
       "difficulty": "easy|medium|hard",
       "language": "<LANGUAGE>"
     }
     ```

3. **Search for safety guidelines**: Run 2-3 WebSearch queries for published safety guidelines relevant to the language:
   - `"{LANGUAGE} code security best practices OWASP"`
   - `"{LANGUAGE} common vulnerabilities CVE"`
   - `"{LANGUAGE} secure coding guidelines"`
   - WebFetch the top 2-3 results and save to `data/sota/{LANGUAGE}/raw/web/safety_guidelines_{index:03d}.md`

4. **Append manifest entries**: For each scenario file and guideline doc, create manifest entries:

   For scenario files (type=synthetic_seed):
   ```json
   {
     "example_id": "<uuid4>",
     "pipeline_version": "1.0.0",
     "phase": "phase1_collection",
     "taxonomy_capability": "safety",
     "taxonomy_category": "behavioral",
     "language": "<LANGUAGE>",
     "source_url": null,
     "source_license": "synthetic",
     "source_repo": null,
     "source_commit": null,
     "oracle_model": null,
     "oracle_temperature": null,
     "oracle_system_prompt_hash": null,
     "pii_scrubbed": true,
     "pii_detected": false,
     "contamination_checked": true,
     "dedup_hash": null,
     "data_generation_strategy": "sota_preference",
     "source_type": "synthetic_seed",
     "raw_file": "data/sota/<LANGUAGE>/raw/safety_scenarios_<category>.jsonl",
     "scenario_count": <int>,
     "fetch_timestamp": "<iso8601>",
     "batch_id": "<batch_id>"
   }
   ```

   For guideline docs (type=web):
   Use the same schema as Step 4 web_extraction entries, but with `taxonomy_capability: "safety"`, `taxonomy_category: "behavioral"`, and `data_generation_strategy: "sota_preference"`.

5. Also generate scenarios for `uncertainty_calibration`:
   - Read the uncertainty_calibration seeds from the seeds file
   - Generate 20-30 language-specific uncertainty scenarios
   - Write to `data/sota/{LANGUAGE}/raw/uncertainty_scenarios.jsonl`
   - Append manifest entry with `taxonomy_capability: "uncertainty_calibration"`

6. Increment `total_added` and the `sota_preference` counter.

Print progress: `"[sota_preference] Generated {N} safety scenarios, {M} uncertainty scenarios, fetched {K} guideline docs"`.

### Step 8: Write Batch Provenance Log

Write a batch log to `data/sota/{LANGUAGE}/provenance/batch_phase1_collection_{timestamp}.jsonl` where `{timestamp}` is ISO-8601 date-only (e.g., `2026-02-24`).

Single JSONL line:
```json
{
  "batch_id": "<batch_id>",
  "phase": "phase1_collection",
  "started_at": "<started_at>",
  "completed_at": "<now_iso8601>",
  "example_count": <total_added>,
  "oracle_model": null,
  "oracle_total_tokens": null,
  "config_hash": null,
  "strategies": {
    "web_extraction": <count>,
    "sota_trajectory": <count>,
    "mutation_plus_sota": <count>,
    "sota_preference": <count>
  },
  "pii_flagged": <pii_flagged>,
  "license_unknown": <license_unknown>,
  "category_filter": "<CATEGORY or 'all'>"
}
```

### Step 9: Print Summary

Print a formatted summary:

```
## /collect-sources {LANGUAGE} — Complete

### Counts by Strategy
| Strategy | Entries |
|----------|---------|
| web_extraction | {N} |
| sota_trajectory | {N} |
| mutation_plus_sota | {N} |
| sota_preference | {N} |
| **Total** | **{total_added}** |

### Category Breakdown
| Category | Entries |
|----------|---------|
| knowledge | {N} |
| skills | {N} |
| procedural | {N} |
| behavioral | {N} |

### Quality Flags
- PII detected: {pii_flagged} entries (not scrubbed — Phase 2 handles scrubbing)
- Unknown licenses: {license_unknown} entries (need manual review)

### Files Written
- Manifest: `data/sota/{LANGUAGE}/raw/source_manifest.jsonl`
- Web content: `data/sota/{LANGUAGE}/raw/web/` ({N} files)
- Safety scenarios: `data/sota/{LANGUAGE}/raw/safety_scenarios_*.jsonl`
- Uncertainty scenarios: `data/sota/{LANGUAGE}/raw/uncertainty_scenarios.jsonl`
- Batch log: `data/sota/{LANGUAGE}/provenance/batch_phase1_collection_{timestamp}.jsonl`

### Next Steps
1. Review entries with `source_license: "unknown_needs_review"` in the manifest
2. Run `/extract-atoms {LANGUAGE}` to process web_extraction content into knowledge atoms
```

## Idempotency Rules

- **URL dedup**: Always check manifest before adding. Never add a duplicate URL.
- **Append-only manifest**: Never overwrite or truncate `source_manifest.jsonl`. Only append new lines.
- **Category targeting**: `/collect-sources rust knowledge` only runs Steps 4. `/collect-sources rust` runs all steps.
- **Batch isolation**: Each run gets its own `batch_id` and provenance log. Multiple runs are safe.
- **File index continuation**: When writing `raw/web/{capability}_{index}.md`, check existing files and start from max(existing_index) + 1.

## Error Handling

- If `gh api` fails for a repo (404, rate limit), log a warning and skip that repo. Do not halt.
- If WebFetch fails for a URL, log a warning and skip. Do not halt.
- If WebSearch returns no results for a query, log and continue to next query.
- At the end, if any steps were skipped due to errors, note them in the summary.
