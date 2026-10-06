unifiedwvalticolocs
===================

Python library that colocates Sentinel-1 WV (wave) OCN Level-2 data with
CMEMS WAVE L3 or CCI SeaState L2P altimeter data, producing NetCDF colocated
products.

.. image:: https://github.com/umr-lops/unifiedwvalticolocs/workflows/build/badge.svg?branch=main
    :target: https://github.com/umr-lops/unifiedwvalticolocs/actions

Colocated products pair each S1 WV OCN measurement with the altimeter
observation (significant wave height) closest in space and time, which is
useful for validation and machine-learning studies of wave models.

Contents
--------

.. toctree::
   :maxdepth: 2

   installation
   usage
   algorithm
   api
