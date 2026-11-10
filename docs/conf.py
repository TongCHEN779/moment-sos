import moment_sos

project = "moment-sos"
author = "moment-sos authors"
release = version = moment_sos.__version__
extensions = ["myst_parser", "sphinx.ext.autodoc", "sphinx.ext.napoleon", "sphinx.ext.viewcode", "sphinx.ext.intersphinx",
              "sphinx_copybutton", "sphinx_design"]
myst_enable_extensions = ["dollarmath", "amsmath", "colon_fence", "deflist"]
myst_heading_anchors = 3
source_suffix = {".md": "markdown", ".rst": "restructuredtext"}
exclude_patterns = ["_build"]
html_theme = "furo"
html_title = "moment-sos"
autodoc_member_order = "bysource"
autodoc_typehints = "description"
napoleon_numpy_docstring = True
intersphinx_mapping = {"python": ("https://docs.python.org/3", None), "numpy": ("https://numpy.org/doc/stable", None),
                       "torch": ("https://pytorch.org/docs/stable", None)}
nitpicky = False
