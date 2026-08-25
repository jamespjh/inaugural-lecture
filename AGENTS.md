# Project Guidelines

## Scope
- Primary codebase is in `demo/`.
- Lecture text and slides at workspace root (`essay.md`),  is narrative content, not executable package code.
- slideshow package is for slide generation, not core simulation logic; it should be treated as a separate component with its own conventions and dependencies, its own virtualenv.

## Architecture
- Two python package root: `demo/src/teachgrav/` and `demo/slideshow/`.
- Entry point for demo code: `teachgrav = teachgrav.entry:entry`.
- Demo code core components:
  - `system.py`: N-body state container and trajectory serialization.
  - `integrator.py`: Euler, SciPy, and diffrax integration methods.
  - `scenarios.py`: ScenarioFactory for `moon`, `sun`, and `scatter` systems.
  - `laws/`: force/acceleration law implementations (`true_law`, `gp`, `pl`).
  - `array_abstraction.py`: backend abstraction for numpy/JAX/MLX arrays.
  - `viz.py`: 2D plotting/animation output.
- Tests mirror package areas under `demo/tests/teachgrav/`.

## Demo Build And Test
Run commands from `demo/` unless task explicitly targets docs/slides only.

1. `python -m venv venv`
2. `source venv/bin/activate`
3. `pip install -e .`
4. `pip install -e '.[dev]'`
5. `pytest`
6. `flake8`

## Virtual environments

When running locally on this laptop in VS Code (not for GitHub `@copilot` tag runs):

Use the existing repository virtual environment at `demo/.venv` for all demo code tasks when running locally.
Do not create additional virtual environments (for example a new root `.venv`) unless the user explicitly asks for that.
Test discovery and running can fail if the sandbox venv is not activated. Always ensure the venv is active before running tests or the demo entry point.
If that fails, explicitly find the executable for pytest or similar in the venv and run it directly, for example `./.venv/bin/pytest`.

## Slide Generation
1. Fill in here
2. In slideshow `.qmd` files, wrap raw HTML (for example `<video>` blocks) inside Pandoc raw blocks using fenced syntax:
  - ` ```{=html}`
  - `<video ...>...</video>`
  - ` ``` `
  This prevents literal HTML text from appearing in rendered slides.

## Conventions
- Use `ScenarioFactory` instead of ad hoc system construction when adding scenario-facing behavior.
- Keep logger name consistent as `Teachgrav` across modules/tests.
- Respect solver/backend compatibility:
  - diffrax methods require a JAX engine (see `entry.py` behavior).
  - default engine is numpy when not otherwise required.
- Preserve data-shape expectations in tests and code:
  - positions/velocities are body-major arrays
  - masses are per-body arrays

## Environment Gotchas
- Visualization/video workflows depend on `ffmpeg` being available on the system path.
- macOS/Darwin has pinned JAX/diffrax dependencies in `demo/pyproject.toml`; do not casually upgrade them without validation.
- `viz.py` is currently 2D-focused; Addition of 3-D visualisation will come later.

## Existing Docs
- Project overview: `demo/Readme.md`
- Lecture manuscript: `essay.md`

## Commit Message Structure
Before running any `git commit`, when working locally rather than as a cloud agent, always show the user in chat:
1. The full proposed commit message (in a code block)
2. The list of files to be staged

Then wait for the user to confirm before executing the commit. Do not commit without explicit user approval.

All commits made with AI assistance must include a structured trailer after the subject line, and a summary of the prompt.


```
<imperative summary of change>

AI-assisted <Model name and harness/tool>.
Prompt summary: "<short summary of the user's instruction>"

AI-Tool: VS Code Copilot agent
AI-Model: <e.g. claude-sonnet-4-5>
AI-Reviewed-By: jamespjh
```

- The subject line uses the imperative mood (e.g. "add X", "fix Y", "refactor Z").
- `AI-Tool` is always `VS Code Copilot agent` for changes made in this environment.
- `AI-Model` should name the model actually used (e.g. `claude-sonnet-4-5`).
- `AI-Reviewed-By` is always `jamespjh`.

- The 'Prompt Summary:' field should be a concise plain-English description of the instruction given to the AI. Neither a verbatim quote nor a copy of the commit message subject line.

## Committing plans

When implementing a plan, add the plan to folder audit/plans in the repo as a markdown file, copyiing the content from plan.md. Name the file for the data and time when implemented, for example `2024-06-20-15-30.md`. This will allow us to keep a record of the plans that were implemented. Use an additional B/C/D etc to resolve any conflicts in the timestamp, for example `2024-06-20-15-30-B.md`.

## Prompt summaries

When making a commit, also add a longer summary of our conversation that led to the commit, in the folder audit/prompts in the repo. Name the file for the data and time when implemented, for example `2024-06-20-15-30.md`. This will allow us to keep a record of the prompts that were used to generate code changes. Use an additional B/C/D etc to resolve any conflicts in the timestamp, for example `2024-06-20-15-30-B.md`. Also include a summary of key decisions and elements of your reasoning.

## Working to close an issue

When working to close an issue:

- work in a branch in the main repo, not a fork
- make commits with the above structure, and push them to the branch
- open a PR from the branch to main, referencing the issue in the PR description
- always write a plan first, and commit the plan to the audit/plans folder with the above structure, before writing any code
- also add the plan as a comment to the issue
- when commenting to issues, when logged in as jamespjh, always include AI-Tool and AI-Model metadata in the comment.
- In the audit/prompts folder, as above, keep a record of the prompts used to generate the code, and a summary of key decisions and elements of your reasoning.
- do not write code before writing a plan
- wait for CI to pass and for jamespjh to review and merge the PR
