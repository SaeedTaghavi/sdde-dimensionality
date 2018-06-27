"""
This code sets up a default environment by importing the packages I most
typically used for this project. Typical use is to type

    from [package].init import *
    %matplotlib inline

in the first cell of a Jupyter notebook. This will print the import code
so as to document which packages are imported. Importing `init` is the same way
as writing all these imports in the first cell documents your namespace; it
just avoids you actually having to type them ;-).
"""

# Determine if we are in IPython session (console or notebook), or normal shell
try:
    get_ipython
except NameError:
    ipython = False
else:
    ipython = True

if ipython:
    # In an interactive session, long output is annoying and can really slow
    # down a notebook
    loglevel = "WARNING"
else:
    # Default level is 'DEBUG'; 'INFO' is a little less verbose
    loglevel = "INFO"

init_code = """
import os
import builtins
from collections import OrderedDict, Iterable, namedtuple, deque
import logging
from multiprocessing import Pool
from zipfile import BadZipFile  # For detecting corrupt files
import numpy as np
from functools import lru_cache, partial
from scipy.special import lambertw
import matplotlib as mpl
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

import luigi
import mlluigi
mlluigi.set_logging_level('{loglevel}')

import theano
from tqdm import tqdm_notebook as tqdm
from parameters import ParameterSet

import theano_shim as shim
import mackelab as ml
import mackelab.iotools
import sinn
from sinn.histories import Series
import sinn.analyze as anlz
import sinn.models

logger = logging.getLogger("linearized_tanh")

plt.style.use('publish')
#plt.style.use('dark_background')
#plt.style.use('default')
plt.rcParams['figure.facecolor'] = (1,1,1,1)
# Use pgf when exporting pdf
from matplotlib.backends.backend_pgf import FigureCanvasPgf
mpl.backend_bases.register_backend('pdf', FigureCanvasPgf)
""".format(loglevel=loglevel)

exec(init_code)
if ipython:
    # Only print in an interactive IPython environment
    print(init_code)
