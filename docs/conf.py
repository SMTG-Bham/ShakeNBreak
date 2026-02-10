# Configuration file for the Sphinx documentation builder.

# -- Path setup --------------------------------------------------------------
# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
import os
import sys

from recommonmark.transform import AutoStructify

sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
project = "shakenbreak"
copyright = "2022, Irea Mosquera-Lois, Seán R. Kavanagh"
author = "Irea Mosquera-Lois, Seán R. Kavanagh"

# The full version, including alpha/beta/rc tags
release = "3.4.4"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.coverage",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosectionlabel",
    "sphinx_click",
    "sphinx_design",
    "myst_nb",  # for jupyter notebooks
    "sphinx_copybutton",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".ipynb": "myst-nb",
}

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "jupyter_execute", "JOSS/paper.md"]
myst_enable_extensions = [
    "html_admonition",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "sphinx_book_theme" # "sphinx_rtd_theme"

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "Images/SnB_logo.png"
html_title = "ShakeNBreak"

# If true, SmartyPants will be used to convert quotes and dashes to
# typographically correct entities.
html_use_smartypants = True

html_theme_options = {
    "repository_url": "https://github.com/SMTG-Bham/ShakeNBreak",
    "repository_branch": "develop",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "home_page_in_toc": True,
    "launch_buttons": {
        "binderhub_url": "https://mybinder.org",
        "colab_url": "https://colab.research.google.com",
    },
}

html_context = {
    "display_github": True, # Integrate GitHub
    "github_user": "SMTG-Bham", # Username
    "github_repo": "ShakeNBreak", # Repo name
    "github_version": "master", # Version
    "conf_py_path": "/docs/", # Path in the checkout to the docs root
}

# -- Options for intersphinx extension ---------------------------------------
# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {
    "pymatgen": ("https://pymatgen.org/", None),
    "doped": ("https://doped.readthedocs.io/en/latest/", None),
}

# -- Options for autodoc -----------------------------------------------------
autoclass_content="both"

# -- Options for nb extension -----------------------------------------------
nb_execution_mode = "off"
nb_render_image_options = {"height": "300",}  # Reduce plots size
#myst_render_markdown_format = "gfm"
myst_heading_anchors = 2
def setup(app):
    app.add_config_value("myst_parser_config", {"auto_toc_tree_section": "Contents"}, True)
    app.add_transform(AutoStructify)

# -- Global substitutions for external links ------------------------------------
# These substitutions are available in all RST files
rst_prolog = """
.. |Structure| replace:: :class:`~pymatgen.core.structure.Structure`
"""