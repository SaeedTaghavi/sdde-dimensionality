## Progress bars

By default, when running in a notebook, we use the progress bars from
tqdm_notebook. These provide better integration with the notebook, but require
that [Jupyter Widgets](https://github.com/jupyter-widgets/ipywidgets) be installed (instructions found [here](https://ipywidgets.readthedocs.io/en/stable/user_install.html)). If they are unavailable, you can change
use the vanilla `tqdm` progress bars; to do so, open `init.py` and change the
line

    'tqdmimport': "from tqdm import tqdm_notebook as tqdm"

to

    'tqdmimport': "from tqdm import tqdm"
