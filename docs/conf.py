#
# svZeroDTrees documentation build configuration file
#
import importlib.metadata
import os
import sys
from datetime import datetime


# Ensure src/ is on the path so autodoc/autoapi can find the package.
sys.path.insert(0, os.path.abspath("../src"))

current_year = datetime.now().year
package_name = "svzerodtrees"


# -- General project information ---------------------------------------------

project = "svZeroDTrees"
copyright = f"Copyright © {current_year} Nick Dorn"
html_show_sphinx = False

try:
    version = importlib.metadata.version(package_name)
except importlib.metadata.PackageNotFoundError:
    version = "0.0.0"

release = version


# -- General configuration ----------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.todo",
    "autoapi.extension",
]

source_suffix = [".rst", ".md"]
master_doc = "index"
pygments_style = "default"
language = "en"


# -- Options for extensions ---------------------------------------------------

myst_enable_extensions = [
    "html_image",
    "colon_fence",
    "deflist",
    "attrs_inline",
]

myst_heading_anchors = 3
myst_footnote_transition = False

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "slides",
    "slides/**",
]


# -- AutoAPI configuration ----------------------------------------------------

autoapi_type = "python"
autoapi_dirs = ["../src/svzerodtrees"]
autoapi_root = "api"
autoapi_add_toctree = False
autoapi_keep_files = False
autoapi_options = ["members", "undoc-members", "show-inheritance"]


# -- Options for HTML output --------------------------------------------------

html_theme = "pydata_sphinx_theme"
htmlhelp_basename = "svzerodtrees_doc"


intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
}
