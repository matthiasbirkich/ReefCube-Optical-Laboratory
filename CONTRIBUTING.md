# Contributing to ReefCube Optical Laboratory

Thank you for your interest in contributing to **ReefCube Optical Laboratory**.

Contributions of all kinds are welcome, including:

- Bug reports
- Documentation improvements
- Example notebooks
- Sensor support
- Calibration methods
- Data analysis functions
- Visualization improvements
- Scientific validation
- New environmental monitoring applications

Our goal is to develop a reliable, transparent and well-documented open-source toolkit for environmental optical measurements and spectroscopy.

---

# Before You Start

For substantial changes or new features, please open a GitHub Issue before beginning implementation. This allows scientific assumptions, software design and implementation details to be discussed before development starts.

Small corrections such as spelling mistakes, documentation improvements or minor bug fixes may be submitted directly as a Pull Request.

---

# Development Setup

## 1. Fork the repository

Create your own fork of the repository on GitHub.

## 2. Clone your fork

```bash
git clone https://github.com/YOUR-USERNAME/ReefCube-Optical-Laboratory.git
cd ReefCube-Optical-Laboratory
```

## 3. Create a feature branch

```bash
git switch -c feature/my-new-feature
```

## 4. Install the package

```bash
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install pytest
```

## 5. Run the test suite

```bash
pytest -v
```

---

# Coding Guidelines

Please follow these recommendations whenever possible.

- Python 3.10 or newer
- Follow PEP 8
- Use type hints for public functions
- Write NumPy-style docstrings
- Keep functions reasonably small
- Prefer readable code over clever code

---

# Documentation

Please update the documentation whenever user-visible behaviour changes.

Public functions, classes and methods should include descriptive docstrings explaining

- purpose
- parameters
- return values
- raised exceptions
- units
- examples (where appropriate)

Scientific calculations should clearly describe assumptions and limitations.

---

# Tests

Whenever practical, new functionality should include automated tests.

Tests should

- be reproducible
- avoid dependencies on local files
- create temporary data when needed
- pass on all supported Python versions

Run the tests before submitting a Pull Request.

```bash
pytest -v
```

GitHub Actions will automatically repeat the tests after every push.

---

# Pull Requests

Before opening a Pull Request, please ensure that

- all tests pass successfully
- documentation has been updated if necessary
- CHANGELOG.md has been updated under the **Unreleased** section
- code has been reviewed

A Pull Request should explain

- what has been changed
- why the change is useful
- how it has been tested

---

# Reporting Bugs

Please use GitHub Issues to report bugs.

Whenever possible, include

- ReefCube Optical Laboratory version
- Python version
- operating system
- minimal reproducible example
- expected behaviour
- observed behaviour
- complete error message

---

# Scientific Contributions

Scientific contributions are particularly welcome.

Examples include

- calibration methods
- spectroscopy
- fluorescence
- reflectance
- irradiance
- PPFD calculations
- uncertainty estimation
- validation datasets

Please document

- measurement conditions
- calibration procedures
- assumptions
- units
- references

whenever applicable.

---

# License

By contributing to this repository, you agree that your contribution will be distributed under the MIT License.
