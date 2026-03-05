# Contributing to splitsmith

Thank you for your interest in contributing to splitsmith! This document provides guidelines for contributing.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/YOUR-USERNAME/splitsmith-v0.1.git`
3. Install in development mode: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature-name`

## Development Workflow

1. Make your changes
2. Add or update tests in `tests/`
3. Run the test suite: `python -m pytest tests/ -v`
4. Ensure all tests pass before submitting

## Code Style

- Follow PEP 8 conventions
- Add docstrings to all public functions
- Use type hints where possible
- Keep functions focused and modular

## Adding a New Split Strategy

1. Add a private `_your_strategy_split()` function in `splitsmith/split.py`
2. Add a dispatch case in the `split()` function
3. Return a `SplitResult` with appropriate metadata
4. Add corresponding tests in `tests/test_split_your_strategy.py`

## Adding a New Audit Check

1. Add a private `_check_your_check()` function in `splitsmith/audit.py`
2. Call it from the `audit()` function
3. Append `Finding` objects with appropriate severity and evidence
4. Add corresponding tests in `tests/test_audit.py`

## Pull Requests

- Provide a clear description of what the PR does
- Reference any related issues
- Ensure all tests pass
- Keep PRs focused on a single change

## Reporting Bugs

Please use the GitHub issue tracker. Include:

- Python version and OS
- Minimal code to reproduce the issue
- Expected vs actual behavior
- Full traceback if applicable

## Feature Requests

Open a GitHub issue with:

- A clear description of the feature
- The use case it addresses
- Any relevant examples or references

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
