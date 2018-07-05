The code in this repository was created in part as experiment for creating
a computational framework combining different tools in the Python ecosystem,
namely [`Luigi`](https://luigi.readthedocs.io/en/stable/index.html),
`functools`'s [`lru_cache`](https://docs.python.org/3/library/functools.html#functools.lru_cache) and [`multiprocessing`](https://docs.python.org/3/library/multiprocessing.html).

I make no apologies for any hacks or clunky design.

# Motivation

This code was created to ease the numerical exploration of stochastic delay
differential equations. It was motivated by the need to compute statistics over
large numbers of stochastic realizations.

# Main features

To facilitate the calculation of empirical statistics for stochastic equations,
this package provides

- Simple definition of new models with a single line of code.
- On-disk caching of system realizations.
- Parallelized generation of realizations.
- Memory-efficient statistics over large numbers of realizations:
  realizations are never in memory all at once, and so can be arbitrarily many.
- Cached computation of statistics to eliminate the cost of keeping
  realizations after the first calculation.
- Parallelized disk access to offset the disk-access cost for the initial
  calculations of statistics.
 
In addition, models can optionally be computed using
[Theano](http://www.deeplearning.net/software/theano/) (see
[below](combining-with-machine-learning-tools)), allowing for
automatic differentiation and translation to C code.

# Installation

For the most part, this package can be installed through `pip`. After navigating
to the local directory to which it was cloned, the usual command

    pip install -e .

will install it along with the required packages. I assume that you will want to
make changes to the code; otherwise the `-e` part of the command may be omitted.

Although not necessary, we recommend installing this package within a [virtual
environment](https://docs.python.org/3/tutorial/venv.html).

## Currently missing dependencies

At present this code requires two packages which are yet public because they
are tied to a manuscript in preparation (these are indicated in `setup.py`.)
You can obtain these by contacting me directly at `alexandre (dot) rene (at) caesar (dot) de`.

## Progress bars

By default, when running in a notebook, we use the progress bars from
`tqdm_notebook`. These provide better integration with the notebook, but require
that [Jupyter Widgets](https://github.com/jupyter-widgets/ipywidgets) be installed
(instructions found [here](https://ipywidgets.readthedocs.io/en/stable/user_install.html)).
If it is unavailable, you can switch to the vanilla `tqdm` progress bars; to do
so, open `init.py` and change the line

    'tqdmimport': "from tqdm import tqdm_notebook as tqdm"

to

    'tqdmimport': "from tqdm import tqdm"

# Combining with machine learning tools

Although models are computed using NumPy by default, they can be
computed with [Theano](http://www.deeplearning.net/software/theano/) instead. 
This allows for a substantial acceleration of simulation time thanks to the automatic
translation and compilation to C code.
Enabling Theano can be done by adding the line

    shim.load_theano()

before instantiating the model. The [`Comparison Theano vs Numpy`](https://github.com/alcrene/sdde-dimensionality/blob/master/notebooks/Comparison%20Theano%20vs%20Numpy.ipynb) notebook demonstrates that we can get 10-fold
speed improvements this way.

Using Theano also makes available a rich library of machine learning tools for
which it forms a basis. One can thus make use of this to perform automatic
differentiation, maximum-likelihood, MCMC sampling, etc.

# Example notebooks

A `notebooks` directory is included, which contain Jupyter notebooks. These
provide example uses of this code base and illustrate some of its features.
Currently there are two:

- [Comparison Theano vs Numpy](https://github.com/alcrene/sdde-dimensionality/blob/master/notebooks/Comparison%20Theano%20vs%20Numpy.ipynb): Shows that enabling Theano can provide
  substantial acceleration when integrating differential equations.
- [Linearized tanh](https://github.com/alcrene/sdde-dimensionality/blob/master/notebooks/Linearized%20tanh.ipynb): Analysis of a nonlinear SDDE with delayed sigmoidal
  feedback. Illustrates the use of the `Realizations` class, along with
  in-memory caching of statistics.
