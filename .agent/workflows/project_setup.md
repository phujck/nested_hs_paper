---
description: Standard directory structure for new research projects.
---

# Project Setup Workflow

When initializing a new research project, create the following directory structure:

```text
project_root/
├── .agent/                 # Agent rules and workflows
├── literature/             # Bibliography and reference papers
│   └── references.bib
├── manuscript/             # LaTeX source
│   ├── build/              # Compilation artifacts
│   ├── tex/
│   │   ├── main.tex        # Driver file
│   │   └── sections/       # Modular content (01_intro, etc.)
├── simulations/            # Code and data
│   ├── src/                # Python scripts
│   ├── results/
│       ├── data/           # Raw simulation output (CSV/JSON)
│       └── figures/        # Generated plots (PDF/PNG)
├── manage.ps1              # Automation script (compilation, plotting)
└── README.md
```

## Steps
1.  **Create Root**: `mkdir <project_name>`
2.  **Scaffold**: Create subdirectories as above.
3.  **Initialize Git**: `git init`
4.  **Create `manage.ps1`**: Copy standard automation script for LaTeX compilation and Python execution.
5.  **Create `main.tex`**: Set up basic RevTeX boilerplate.
