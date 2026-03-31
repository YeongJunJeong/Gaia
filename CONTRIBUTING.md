# Contributing to Gaia

Thank you for your interest in contributing to Gaia! This document provides guidelines for contributing to the project.

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/your-username/gaia.git`
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate`
4. Install in development mode: `pip install -e ".[dev]"`
5. Create a branch: `git checkout -b feature/your-feature-name`

## Types of Contributions

### Code Contributions

- Bug fixes and improvements
- New preprocessing modules
- Model architecture enhancements
- Evaluation metrics and benchmarks
- Documentation improvements

### Data Contributions

- New soil microbiome datasets (must follow our [data standard](docs/data_standard.md))
- Metadata must include at minimum: biome type and geographic location
- Run the quality validation script before submitting: `python data/scripts/validate_quality.py your_data.csv`

### Scientific Contributions

- New benchmark task proposals
- Ecological interpretation and validation
- Domain expert review of model outputs

## Development Guidelines

### Code Style

- We use **Black** for code formatting (line length: 88)
- We use **isort** for import sorting
- We use **flake8** for linting
- Run all formatters before committing:

```bash
black gaia/ tests/
isort gaia/ tests/
flake8 gaia/ tests/
```

### Testing

- Write tests for all new functionality
- Tests go in the `tests/` directory, mirroring the `gaia/` structure
- Run tests with: `pytest tests/`
- Ensure all tests pass before submitting a PR

### Commit Messages

- Use clear, descriptive commit messages
- Format: `type: short description`
- Types: `feat`, `fix`, `docs`, `test`, `refactor`, `chore`
- Example: `feat: add CLR normalization to preprocessing pipeline`

## Pull Request Process

1. Update documentation if you changed any interfaces
2. Add tests for new functionality
3. Ensure all CI checks pass
4. Request review from at least one maintainer
5. PRs require one approval before merging

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include: steps to reproduce, expected behavior, actual behavior
- For data-related issues, include sample size and data source

## Code of Conduct

- Be respectful and inclusive
- Focus on constructive feedback
- Help newcomers get started

## Questions?

- Open a GitHub Discussion for general questions
- Join our Discord server for real-time chat
- Attend monthly community meetings (1st Thursday of each month)

## License

By contributing, you agree that your contributions will be licensed under the Apache License 2.0.
