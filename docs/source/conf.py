# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import pathlib
import sys
import os
import mav_sim
import shutil
sys.path.insert(0, pathlib.Path(__file__).parents[3].resolve().as_posix())
sys.path.insert(0, os.path.abspath('.'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'mav_sim'
copyright = '2026, Greg Droge'
author = 'Greg Droge'
release = '26'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ['sphinx.ext.duration',    # Times the creation of the build
              'sphinx.ext.autodoc',     # Allows for autodocumentation
              'sphinx.ext.autosummary', # Allows for creation of the toc
              'myst_parser',            # Allows for use of markdown files (md)
              'sphinx.ext.napoleon',    # Allows for the use of google-style documentation
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
