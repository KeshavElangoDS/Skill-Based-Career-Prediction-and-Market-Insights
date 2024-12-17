# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'Skill-Based Career Prediction and Market Insights Using Machine Learning and Hollands Code'
copyright = '2024, Aditya Birla, Keshav Elango, Raman Srinivas Naik (Co-Author)'
author = 'Aditya Birla, Keshav Elango, Raman Srinivas Naik (Co-Author)'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

import os
import sys

sys.path.insert(0, os.path.abspath('..'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
]

templates_path = ['_templates']
exclude_patterns = []

source_encoding = 'utf-8-sig'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']

html_baseurl = 'https://keshavelangods.github.io/Skill-Based-Career-Prediction-and-Market-Insights/'
html_theme = 'sphinx_rtd_theme'
html_title = "Skill-Based Career Prediction and Market Insights Using Machine Learning and Hollands Code Documentation"