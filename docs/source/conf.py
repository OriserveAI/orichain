import os
import sys

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Orichain"
copyright = "2025, Oriserve"
author = "Apoorv Singh, Shubham Maindola"
release = "2.4.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []
templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_show_sourcelink = False
html_theme = "furo"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_theme_options = {
    "light_logo": "logo-light.png",
    "dark_logo": "logo-dark.png",
}


# -- Enabling sphinx.ext.autodoc (and others) to pull docstrings automatically -------------------------------------------------

extensions = ["sphinx.ext.autodoc", "sphinx.ext.viewcode"]
sys.path.insert(0, os.path.abspath("../../src"))
autodoc_member_order = "bysource"
