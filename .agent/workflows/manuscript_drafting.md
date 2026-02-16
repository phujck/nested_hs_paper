---
description: Workflow for drafting, polishing, and submitting manuscripts.
---

# Manuscript Workflow

## 1. Planning
- detailed task lists in `task.md`.
- Implementation plans for major sections.

## 2. Drafting (`manuscript/tex/sections/`)
- Write in modular files: `01_introduction.tex`, `02_model.tex`, etc.
- Use `[ref]`, `[cite]`, or `[figure]` placeholders to maintain flow.
- Focus on content first, formatting second.

## 3. Iterative Refinement
- **Review**: User reviews generated text.
- **Edit**: Agent modifies specific section files.
- **Compile**: Run `./manage.ps1 paper` to check PDF output.

## 4. Polishing (The "Polish Pass")
- **Spell Check**: Convert all text to **British English**.
- **Citations**: Resolve all `[ref]` placeholders. Ensure key authors (Anders, Ankerhold, Nazir) are cited.
- **Formatting**: Fix floats, overfull hboxes, and equation numbering.

## 5. Submission (`arxiv_submission_flat/`)
- **Automated Flattening**:
    - Run the standardized script: `powershell -ExecutionPolicy Bypass -File .agent/scripts/flatten_arxiv.ps1`
    - This script automates:
        - Cleaning `arxiv_submission_flat/`
        - Copying and flattening all assets
        - Rewriting `\input` and `\includegraphics` paths
        - Creating `arxiv_submission_flat.zip`
- **Manual Verification**:
    - Unzip and check `main.tex` for correct paths.
    - Upload `arxiv_submission_flat.zip` to arXiv.
