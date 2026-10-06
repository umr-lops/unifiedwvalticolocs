# AGENTS.md

Python 3.11+ library that colocates Sentinel-1 WV (wave) OCN Level-2 data with CMEMS WAVE L3 or CCI SeaState L2P altimeter data, producing NetCDF coloc products. Primary target environment: IFREMER HPC (PBS/SLURM, Apptainer).

## Commands

```bash
pip install -e ".[dev]"          # editable install + dev tools (use a venv first)
pre-commit install               # then run pre-commit hooks
make test                        # pytest + coverage; REGENERATES assets/images/coverage.svg (expect a modified file)
make check-codestyle             # isort + black + flake8 + darglint
make mypy                        # strict mypy (config in pyproject.toml)
make check-safety                # bandit
make lint                        # test + check-codestyle + mypy + check-safety
make codestyle                   # auto-fix formatting (pyupgrade, isort, black)
```

- CI (`.github/workflows/build.yml`) runs only: `check-codestyle`, `test`, `check-safety` — it does NOT run mypy. Run `make mypy` locally before pushing.
- Single test: `PYTHONPATH=$(pwd) pytest tests/test_core_coloc.py -k test_name -c pyproject.toml`
- pytest is configured with `--doctest-modules`: doctests in `unifiedwvalticolocs/*.py` run as part of `pytest`.

## Code conventions that bite

- Docstrings must be Google-style and complete: `darglint` runs with `strictness = long` (setup.cfg) — undocumented args/returns fail `make check-codestyle`.
- flake8 ignores E203/E501/W503 (setup.cfg); black line length is 88.
- `mypy --install-types --non-interactive` is part of `make install`; new third-party imports may require stubs.

## Layout

- `unifiedwvalticolocs/unified_coloc_WV_alti_cmems_or_cci.py` — the whole pipeline: `core_coloc()` + CLI `entrypoint()` (~1200 lines, steps `step_0` … `step_3`).
- `unifiedwvalticolocs/create_listing_jobarray.py` — builds CSV job-array listings for HPC (CLI: `create-unified-wv-alti-job-array-listing`, `--infra ice|hpc`).
- `unifiedwvalticolocs/unified_coloc_WV_alti_cmems_or_cci_pbs`, `..._slurm.bash`, `..._prun.py` — HPC glue (PBS/SLURM/prun submission, Apptainer `.sif` images). `_slurm.bash` is dual-mode: single-shot (`--startdate ...`) or job array (`--listing CSV`, one array task per row). `submit_slurm_jobarray.sh` builds the listing and submits the array. They contain hardcoded HPC paths (`/scale/...`, `/appli/...`) and legacy `python2.7` shebangs; do not "fix" these casually, they target specific machines.
- `unifiedwvalticolocs/config.yml` — sample runtime config with HPC data paths; the CLI `--config` arg points to a real config at runtime. `localconfig_new_storage.yml` is a local variant.
- `unifiedwvalticolocs/data4tests/` — small SAR NetCDF fixture used by tests; keep tests self-contained with synthetic `xarray` data + this fixture, no network access.

## Packaging / release

- Version is derived from git tags via hatch-vcs — there is no version field to edit anywhere. Release = `git tag <semver>`, push tag, `python -m build`, `twine upload dist/*`.
- `_version.py` is generated at build time; `_version.pyi` is the checked-in stub.

## Git

- Default branch is `main` (README template text mentioning `master` is stale). Never push directly to `main`; work on feature branches and open PRs.
