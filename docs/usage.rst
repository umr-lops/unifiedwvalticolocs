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

Sample configs are shipped in the package: ``config.yml`` (HPC paths) and
``localconfig_new_storage.yml``.

HPC job arrays
--------------

For production runs, generate a listing and submit it as a PBS/SLURM job
array (see the ``.pbs`` / ``_slurm.bash`` / ``_prun.py`` scripts in the
package):

.. code-block:: bash

   create-unified-wv-alti-job-array-listing \
       --infra ice \
       --outputpath-csv /path/to/listing.csv \
       --start 20260101 --stop 20260131
