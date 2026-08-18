# Paper Correction and GitHub Publication Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a corrected author-copy paper and publish a reproducible Phase D research repository.

**Architecture:** A deterministic PDF correction script replaces one figure through a vector overlay. Repository curation is enforced by `.gitignore`; the README and research note connect the paper, corrected protocol, measured results, and prioritized future experiments.

**Tech Stack:** Python 3.13, pypdf, reportlab, TSFEL, scikit-learn, pytest, Git, GitHub CLI.

## Global Constraints

- Preserve the official source PDF and label the edited PDF as an author-corrected copy.
- Do not commit raw participant data, caches, temporary files, or regenerated search artifacts.
- Do not claim future work as implemented or validated.
- Publish only after tests, PDF render inspection, and artifact integrity checks pass.

---

### Task 1: Correct Figure 1 Reproducibly

**Files:**
- Create: `tools/fix_paper_figure1.py`
- Create: `paper/Ly_2026_J_Phys_Conf_Ser_3180_012004_corrected_author_copy.pdf`
- Create: `paper/README.md`

- [ ] Implement a reportlab vector overlay for PDF page 4 with seven non-duplicated stages.
- [ ] Merge the overlay with pypdf without modifying any other page.
- [ ] Render all pages and visually inspect page 4 plus page-count and text integrity.

### Task 2: Curate the GitHub Repository

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Modify: `README.md`
- Create: `docs/PAPER_AND_FUTURE_WORK.md`

- [ ] Exclude data, caches, temporary directories, intermediate checkpoints, and large generated tables.
- [ ] Keep final models, compact result CSV/JSON files, notebooks, tests, and corrected paper.
- [ ] Document installation, reproduction, paper citation, measured results, limitations, and future experiments.

### Task 3: Verify and Publish

**Files:**
- Verify: `tests/`
- Verify: notebooks, models, S4 submission, corrected PDF

- [ ] Run the full pytest suite and compile repository Python files.
- [ ] Check notebook cell errors, model round trips, submission schema, and PDF rendering.
- [ ] Initialize Git, stage the curated scope, inspect staged size, and commit.
- [ ] Refresh GitHub authentication, create `LCT-dotcom/keypoint-activity-recognition-phase-d`, and push `main`.
