# Writing Style Guidelines

## Tone and Voice
- **Academic & Authoritative**: Use clear, precise language. Avoid colloquialisms but maintain a natural flow.
- **Humble but Confident**: Acknowledge limitations ("hardly novel") while asserting the value of the contribution ("rigorous bridge").
- **Concise**: Avoid unrelated fluff. Every sentence should advance the argument.

## Spelling and Grammar
- **British English**: Always use British spelling conventions.
    - `centre` (not center)
    - `colour` (not color)
    - `realise`, `generalise` (not realize, generalize)
    - `modelling` (not modeling)
    - `behaviour` (not behavior)
    - `analyse` (not analyze)
    - `programme` (not program, except for computer code)

## Formatting (LaTeX/RevTeX)
- **Document Class**: `\documentclass[aps,prl,twocolumn,superscriptaddress,showpacs,floatfix]{revtex4-2}` (or similar).
- **Floats**: Use `[htbp]` to allow LaTeX to place figures optimally.
- **Equations**: Number meaningful equations. Use `align` environments for multi-line derivations.
- **References**: Use `\cite{key}`. Ensure `[ref]` placeholders are resolved before polishing.

## Citations
- **Key Authors**: Prioritize citing:
    - **Janet Anders**: Thermodynamics, strong coupling.
    - **Joachim Ankerhold**: Path integrals, quantum dynamics.
    - **Ahsan Nazir**: Reaction coordinates, polaron mappings.
- **Bibliography**: Maintain a consolidated `.bib` file (e.g., `literature/references_new.bib`).

## Manuscript Structure
- **Modular Sections**: Break the manuscript into separate `.tex` files (e.g., `sections/01_introduction.tex`).
- **Main Driver**: `main.tex` should primarily contain preamble and `\input{}` commands.
