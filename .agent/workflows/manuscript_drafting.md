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

## 5. Submission (`arxiv_submission/`)
- Create a flattened directory.
- Copy `main.tex` and `main.bbl`.
- Copy all section files.
- Copy all figures to a local `figures/` folder.
- Update figure paths in `.tex` files to point to local `figures/`.
- Zip the folder for upload.
