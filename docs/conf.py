"""Sphinx configuration for unifiedwvalticolocs."""

import os
import sys
from importlib import metadata as importlib_metadata

sys.path.insert(0, os.path.abspath(".."))

try:
    release = importlib_metadata.version("unifiedwvalticolocs")
except importlib_metadata.PackageNotFoundError:
    release = "unknown"

project = "unifiedwvalticolocs"
author = "umr-lops"
copyright = "2026, umr-lops"

version = release.split(".")[0]
html_theme = "sphinx_rtd_theme"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

napoleon_google_docstring = True
autodoc_typehints = "description"

html_static_path = []
html_title = project
html_logo = None

exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
