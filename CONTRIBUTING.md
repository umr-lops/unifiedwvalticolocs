# How to contribute

## Dependencies

We use standard `pip` with a `pyproject.toml` (built with [Hatchling](https://hatch.pypa.io/latest/build/)) to manage dependencies.

To install dependencies and prepare [`pre-commit`](https://pre-commit.com/) hooks you would need to run `install` command:

```bash
make install
make pre-commit-install
```

`make install` installs the project in editable mode with its dev dependencies (`pip install -e ".[dev]"`) into whatever Python environment is currently active — create and activate a virtual environment first (`python -m venv .venv && source .venv/bin/activate`, or `micromamba create`/`conda create`) if you haven't already.

## Codestyle

After installation you may execute code formatting.

```bash
make codestyle
```

### Checks

Many checks are configured for this project. Command `make check-codestyle` will check black, isort and darglint.
The `make check-safety` command will look at the security of your code.

Comand `make lint` applies all checks.

### Before submitting

Before submitting your code please do the following steps:

1. Add any changes you want
1. Add tests for the new changes
1. Edit documentation if you have changed something significant
1. Run `make codestyle` to format your changes.
1. Run `make lint` to ensure that types, security and docstrings are okay.

## Other help

You can contribute by spreading a word about this library.
It would also be a huge contribution to write
a short article on how you are using this project.
You can also share your best practices with us.
