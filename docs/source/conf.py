# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information


import os
import sys
import pathlib

# path = pathlib.Path(__file__).resolve() / '..' / '..'

sys.path.insert(0, os.path.abspath(".."))
# sys.path.insert(0, os.path.abspath('../..'))
# sys.path.insert(0, os.path.abspath(path))

project = 'HyperGP'
copyright = '2024, ZhitongMa'
author = 'ZhitongMa'
release = 'v1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'recommonmark',
    'sphinx_markdown_tables',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.intersphinx',
    # 'sphinx.ext.linkcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx_design',

    'sphinx.ext.doctest',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
]

templates_path = ['_templates']
exclude_patterns = []

language = 'zh_CN'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# html_theme = 'alabaster'
# html_theme = 'classic'
# html_theme = 'sphinx_rtd_theme'
# html_theme = 'pydata_sphinx_theme'
# html_theme = 'press'
html_theme = 'sphinx_book_theme'
html_static_path = ['_static']

html_theme_options = {
    # 'show_toc_level': 2,
    # 'repository_url': 'https://github.com/jax-ml/jax',
    # 'use_repository_button': True,     # add a "link to repository" button
    # 'navigation_with_keys': False,
}

autodoc_typehints_format = "short"
autodoc_typehints = "description"
autodoc_class_signature = "separated"

autodoc_typehints_description_target = "all"

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = False
numpydoc_show_class_members = False
autosummary_imported_members = True
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath"]
# locale_dirs = ["locale/"]
gettext_compact = "docs"
add_module_names = False