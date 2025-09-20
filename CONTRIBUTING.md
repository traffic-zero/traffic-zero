# Contributing to Traffic Zero

Thanks for your interest in contributing! This document outlines how to collaborate efficiently on this repository.

## How to Contribute

1. **Fork the Repository**
   Click the “Fork” button on GitHub to create your own copy.

2. **Clone Your Fork**
    ```bash
    git clone git@github.com:<your-username>/traffic-zero.git
    cd traffic-zero
    ```

3. **Create a Branch**
    ```bash
    git checkout -b feature/your-feature-name
    ```

    Prefix branches based on type:

    - `sim/` → Simulation or environment changes
    - `agent/` → Controller or RL agent
    - `data/` → Dataset scripts, preprocessing
    - `experiment/` → Config or experiment pipeline changes
    - `dashboard/` → Visualization / dashboard improvements
    - `docs/` → Documentation updates


4. **Make Changes**
   Implement your feature or fix the bug. Ensure code quality and consistency.

5. **Test Your Changes**
   Run existing tests and add new ones if applicable.

6. **Commit Your Changes**
    ```bash
    git add .
    git commit -m "Add your descriptive commit message"
    ```

    Please adhere to [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) style for commit messages.

7. **Push to Your Fork**
    ```bash
    git push --set-upstream origin feature/your-feature-name
    ```

8. **Create a Pull Request**
    Go to your fork on GitHub and click the “Compare & pull request” button. Provide a clear description of your changes. Address feedback from maintainers.

    Ensure PR description explains:

    - Purpose of change
    - Files modified
    - Any testing performed
    - Relevant issue number (if applicable)

## Guidelines
- **Commit size**: Use small, focused commits.
- **Code Style**: Follow PEP 8 for Python code. Use linters & formatters like `flake8` or `black`. (Recommended: set up pre-commit hooks using `pre-commit install`.)
- **Documentation**: Update documentation for new features or changes. Use docstrings for functions and classes.
- **Experiments**:
    - Log seeds, parameters, and environment configuration.
    - Provide reproducible scripts & configs to run experiments.
- **Data Privacy**:
    - Do not include sensitive or proprietary data in the repository.
    - Do not include large raw datasets in the repo; use scripts to download/process.
- **Testing**: Write tests for new features and bug fixes. Ensure all tests pass before submitting a PR.
- **Code Reviews**: Every PR must be reviewed by at least one other team member before merging.

## Communication
- Use GitHub issues for bugs, feature requests, and discussions.
- Tag issues with relevant labels (bug, enhancement, question).
- For major changes, discuss via issues or team meetings before implementation.

## Reporting Issues
If you encounter bugs or have feature requests, please open an issue on GitHub with detailed information. Check existing issues to avoid duplicates.

## Code of Conduct
This project adheres to a [Contributor Covenant Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/). By participating, you agree to abide by its terms to foster a positive and collaborative environment.
