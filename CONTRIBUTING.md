# Contributing to nanoDist

First off, thank you for considering contributing to nanoDist. It's people like you that make nanoDist such a great tool.

## Where do I go from here?

If you've noticed a bug or have a feature request, make one! It's generally best if you get confirmation of your bug or approval for your feature request this way before starting to code.

## Fork & create a branch

If this is something you think you can fix, then fork nanoDist and create a branch with a descriptive name.

A good branch name would be (where issue #325 is the ticket you're working on):

```sh
git checkout -b feature/325-add-new-optimizer
```

## Implementation guidelines

- **Coding Standard**: We adhere strictly to PEP-8. All code must pass `ruff` linting and formatting.
- **Type Checking**: All functions must have Python type hints and pass `mypy` static type checking.
- **Testing**: We use `pytest`. Please add tests for any new functionality and ensure all existing tests pass before submitting a PR.
- **Documentation**: All new features or architectural changes must be updated in `README.md`.

## Make a Pull Request

At this point, you should switch back to your master branch and make sure it's up to date with nanoDist's master branch:

```sh
git remote add upstream git@github.com:PundarikakshNTripathi/nanoDist.git
git fetch upstream
git rebase upstream/main
```

Then push your branch to GitHub and submit a Pull Request!
