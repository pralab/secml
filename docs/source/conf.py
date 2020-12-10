# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# http://www.sphinx-doc.org/en/master/config

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'SecML'
import datetime
copyright = '{:}, PRALab - Pattern Recognition and Applications Lab & ' \
            'Pluribus One s.r.l.'.format(datetime.datetime.now().year)
author = 'PRALab'

# The full version, including alpha/beta/rc tags
import secml
release = version = secml.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.imgmath',
    'matplotlib.sphinxext.plot_directive',
    'numpydoc',
    'm2r',
    'nbsphinx',
    'nbsphinx_link'
]

autodoc_default_options = {
    # 'members': True,
    'member-order': 'alphabetical',
    # 'undoc-members': True,
    # 'show-inheritance': True,
    'exclude-members': ''
}

# The following modules should be faked by sphinx (e.g. extras)
autodoc_mock_imports = [
    "pytest", "torch", "torchvision", "cleverhans", "tensorflow"]

# Autosummary pages will be generated by sphinx-autogen instead of sphinx-build
# autosummary_generate = True

numpydoc_class_members_toctree = False

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Always show the source code that generates a plot
plot_include_source = True
plot_formats = ['png']

source_suffix = ['.rst', '.md']

img_latex_preamble = r'\\usepackage{amsmath}'

intersphinx_mapping = {
    'python': ('https://docs.python.org/{.major}'.format(
        sys.version_info), None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference', None),
    'sklearn': ("https://scikit-learn.org/stable/", None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pytorch': ('https://pytorch.org/docs/stable/', None),
    'cleverhans': ('https://cleverhans.readthedocs.io/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------

add_function_parentheses = False
html_show_sourcelink = False
html_show_sphinx = True
html_show_copyright = True

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# html_logo = '_static/secml.png'
# html_favicon = '_static/favicon.png'

html_theme_options = {
    'canonical_url': '',
    'analytics_id': 'UA-145155489-1',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980B9',
    # Toc options
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
