Installation
============

Requirements
------------

- Python 3.11+
- `xarray`, `pandas`, `tqdm`, `scipy`, `h5netcdf`, `h5py`, `dask`, `pyyaml`
  (installed automatically)

From PyPI
---------

.. code-block:: bash

   pip install -U unifiedwvalticolocs

or with `uv`:

.. code-block:: bash

   uv pip install -U unifiedwvalticolocs

From source (development)
-------------------------

.. code-block:: bash

   git clone https://github.com/umr-lops/unifiedwvalticolocs.git
   cd unifiedwvalticolocs
   python -m venv .venv
   source .venv/bin/activate
   pip install -e ".[dev]"
   pre-commit install

The version is derived from git tags by `hatch-vcs`; an editable install
from an untagged checkout gets a dev version.

Visualization extra
-------------------

The colocation illustration helpers (SAFE footprints + WV imagette +
altimeter track maps) need optional plotting dependencies:

.. code-block:: bash

   pip install -U "unifiedwvalticolocs[viz]"

or from an editable install:

.. code-block:: bash

   pip install -e ".[viz]"

This installs ``cartopy``, ``matplotlib`` and ``shapely``.

HPC / Apptainer
---------------

The target environment is IFREMER HPC (PBS/SLURM, Apptainer). An Apptainer
definition file is provided at the repository root:

.. code-block:: bash

   apptainer build unifiedwvalticolocs.sif apptainer.def
   apptainer run unifiedwvalticolocs.sif [args...]
