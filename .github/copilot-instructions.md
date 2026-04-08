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

## Slide Generation
1. Fill in here

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
All commits made with AI assistance must include a structured trailer after the subject line:

```
<imperative summary of change>

AI-assisted <Model name and harness/tool>.
Prompt: "<short summary of the user's instruction>"

AI-Tool: VS Code Copilot agent
AI-Model: <e.g. claude-sonnet-4-5>
AI-Reviewed-By: jamespjh
```

- The subject line uses the imperative mood (e.g. "add X", "fix Y", "refactor Z").
- `AI-Tool` is always `VS Code Copilot agent` for changes made in this environment.
- `AI-Model` should name the model actually used (e.g. `claude-sonnet-4-5`).
- `AI-Reviewed-By` is always `jamespjh`.
- The `Prompt:` field is a concise plain-English summary of what was asked, not a verbatim quote.
