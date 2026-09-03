# Fixed Resource Communication

This repository contains a small, deterministic multi-role communication
prototype. Core modules live under `src/modules/`, tests under `tests/`, and
configuration under `configs/`.

## Install

```bash
python -m pip install -r requirements.txt
```

## Run the demo

Windows: `run_demo.bat`

Linux/macOS: `python scripts/run_demo.py`

The demo prints key shapes and values for the complete forward pipeline.

## Run all tests

```bash
python -m pytest
```

The suite covers structure, profile constraints, math functions, LP, coding,
and channel behavior.

## Run the complete Toy acceptance flow

```bash
python run_toy.py
```

This command validates all 13 profiles, runs the random-tensor forward demo,
executes structure tests, and then runs the math Toy and LP constraint tests.
It exits with a non-zero status when any stage fails and prints
`All Toy Tests PASSED` after a successful run.
