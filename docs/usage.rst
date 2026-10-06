Usage
=====

Command-line entry points
-------------------------

Two executables are installed:

- ``procunifiedwvalticolocs`` — run the colocation for one day / one satellite pair.
- ``create-unified-wv-alti-job-array-listing`` — build the CSV job-array
  listing used to submit batch jobs on HPC.

Colocate one day
~~~~~~~~~~~~~~~~

.. code-block:: bash

   procunifiedwvalticolocs \
       --outputdir /path/to/output \
       --startdate 20260112 \
       --sat S1A \
       --alt cmems_Jason-3 \
       --config /path/to/config.yml

Options:

.. list-table::
   :header-rows: 1

   * - Option
     - Description
   * - ``--outputdir``
     - Folder where the colocated NetCDF is written (required).
   * - ``--startdate``
     - Day to analyze, ``YYYYMMDD`` (required).
   * - ``--sat``
     - SAR mission, e.g. ``S1A`` or ``S1B`` (required).
   * - ``--alt``
     - Altimeter, ``cmems_<mission>`` or ``cci_<mission>`` (required).
       CMEMS: SARAL, cryosat-2, CFOSAT, Jason-3, Sentinel-3A/B, HY2B, HY2C,
       Sentinel-6A, SWOT-Nadir. CCI: cryosat-2, jason-2, jason-3,
       sentinel-3a/b, saral, sentinel-6.
   * - ``--config``
     - Path to the YAML config file (required).
   * - ``--redo``
     - Redo files that already exist (default: skip).
   * - ``--progressbar``
     - Show tqdm progress bar.
   * - ``--dev``
     - Quick run for development/testing (stops after a few matchups).
   * - ``--verbose``
     - Debug logging.

The output file is written to
``<outputdir>/<sat>_<alt>/<year>/coloc_<date>_<sat>_WV_<alt>_<dt>_min_<dist>_km.nc``
alongside a ``.lst`` file listing the altimeter files used for each
colocated WV measurement.

Configuration file
------------------

The config file (YAML) provides the data locations and matching thresholds:

.. code-block:: yaml

   DIR_OUT_ROOT: "/path/to/output/root"
   path_SAR: "/path/to/esa/s1/l2/wv/"
   DIR_OUT_SUBDIRS: "analysis/s1_data_analysis/hs_nn/unified_colocs_wv_alti"
   subset_alti_name_dir: "cmems_obs-wave_glo_phy-swh_nrt_%s-l3_PT1S"
   cmems_dir: "/path/to/cmems/wave/WAVE_GLO_PHY_SWH_L3_NRT_014_001/"
   cci_alti_dir: "/path/to/cci/seastate/v5/"
   delta_dist_km: 30
   delta_t_minutes: 25

- ``path_SAR``: root of the ESA S1 Level-2 ``WV/OCN`` tree
  (``<sat>_WV_OCN__2S/<year>/<doy>/*.SAFE``).
- ``cmems_dir`` / ``cci_alti_dir``: roots of the altimeter datasets.
- ``delta_dist_km``: maximum SAR–altimeter distance (km).
- ``delta_t_minutes``: maximum SAR–altimeter time gap (minutes).

A sample config is shipped in the package (``config.yml``, HPC paths); copy
it and adjust the paths and matching thresholds for your environment.

HPC job arrays
--------------

For production runs, generate a listing and submit it as a PBS or SLURM job
array. Without ``--start``/``--stop`` each SAR x altimeter pair defaults to
the intersection of its acquisition windows (pairs with no overlap are
skipped).

SLURM: submit the listing with
`turboblast <https://github.com/umr-lops/turboblast>`_ (``pip install
turboblast``), which chunks large listings to respect the cluster's array
limits:

.. code-block:: bash

   create-unified-wv-alti-job-array-listing \
       --infra hpc \
       --outputpath-csv /scratch/$USER/listing.txt \
       --output-type txt \
       --sar-units S1A --alt cmems_Jason-3

   SLURM_SCRIPT=$(python -c 'import os, unifiedwvalticolocs as u; print(os.path.join(os.path.dirname(u.__file__), "unified_coloc_WV_alti_cmems_or_cci_slurm.bash"))')

   turboblaster \
       --listing-input /scratch/$USER/listing.txt \
       --bash-slurm-exec "$SLURM_SCRIPT" \
       --slurm-partition cpu --mem 5G --timeout-min 1180

``turboblaster`` blocks until all tasks finish, with a live progress bar
(run it under ``tmux``/``nohup`` for multi-day runs).

The PBS (``.pbs``) and prun (``_prun.py``) launchers consume the same
listing format, e.g.:

.. code-block:: bash

   create-unified-wv-alti-job-array-listing \
       --infra ice \
       --outputpath-csv /path/to/listing.txt \
       --output-type txt \
       --start 20260101 --stop 20260131
