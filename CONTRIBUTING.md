# Contributing to Axiom-OS

Thank you for your interest in contributing to Axiom-OS! This document provides guidelines for contributing.

## Code of Conduct

Please be respectful and professional. We are a scientific research community.

## How to Contribute

### Reporting Bugs

1. Check if the issue already exists
2. Create a detailed bug report with:
   - Clear title
   - Steps to reproduce
   - Expected vs actual behavior
   - Environment details (Python version, OS, PyTorch version)

### Suggesting Features

1. Open an issue with the `enhancement` tag
2. Describe the feature and its use case
3. Explain how it fits into the existing architecture

### Pull Requests

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Follow the code style (Black + Ruff)
4. Write tests for new functionality
5. Ensure all tests pass
6. Commit with clear messages
7. Push to your fork and submit a PR

## Development Setup

```bash
# Clone the repository
git clone https://github.com/yuzechen83-crypto/axiom-os.git
cd axiom-os

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate  # Windows

# Install with dev dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

- We use **Black** for formatting (line-length: 100)
- We use **Ruff** for linting
- We use **Type hints** where possible
- All functions should have docstrings

## Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_example.py

# Run with coverage
pytest --cov=axiom_os --cov-report=html
```

## Project Structure

```
axiom_os/
├── core/           # Core physics modules
├── layers/         # Neural network layers
├── neurons/        # RCLN neurons
├── datasets/       # Data loaders
├── experiments/    # Experiment scripts
├── benchmarks/     # Benchmark utilities
├── agent/          # Agent/Chat UI
└── tests/          # Test suite
```

## License

By contributing, you agree that your contributions will be licensed under the MIT License.
