# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
from datetime import datetime

# Add project src/ to sys.path so autodoc can import project packages
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'HELIOS'
author = 'Vincent Foriel'
copyright = f'{datetime.now().year}, {author}'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_parser',
    'sphinx_copybutton',
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '.venv']

# -- Autodoc configuration ---------------------------------------------------

autodoc_member_order = 'bysource'
autodoc_typehints = 'signature'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# Mock imports for modules that may not be available during doc build
autodoc_mock_imports = [
    'matplotlib',
    'scipy',
    'astropy',
    'numpy',
]

# Napoleon settings: support both Google and NumPy docstring styles
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- MyST-Parser configuration -----------------------------------------------

# Enable MyST extensions for richer markdown features
myst_enable_extensions = [
    'amsmath',
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'replacements',
    'smartquotes',
    'strikethrough',
    'substitution',
    'tasklist',
]

# Allow fenced code blocks to act as Sphinx directives
myst_fence_as_directive = [
    'autoclass',
    'automodule',
    'autofunction',
    'autodata',
    'autoexception',
    'autosummary',
]

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'breeze'
html_title = 'HELIOS'
html_static_path = ['_static']

html_context = {
    "github_user": "VForiel",
    "github_repo": "HELIOS",
    "github_version": "main",
    "doc_path": "docs",
}

html_theme_options = {
    "emojis_header_nav": True,
}
