from init import *
import io
import inspect
#from enum import Enum
from models import TanhModel as DefaultModel

Defaults = namedtuple('Defaults', ['seeds', 'datadir', 'lag_dt', 'cores'])
defaults = Defaults(
    seeds = list(range(10)),
    datadir = 'data',
    lag_dt = 0.01,
    cores = 3
    )

def getdefault(parameter):
    return inspect.signature(DefaultModel).parameters[parameter].default

class NoWarningParameter(luigi.Parameter):
    # Workaround to disable warning on Model parameter until
    # we write our own Parameter class
    def _warn_on_wrong_param_type(self, param_name, param_value):
        return

default_params = DefaultModel.Parameters()
class Task(luigi.Task):
    Model = NoWarningParameter(default=DefaultModel)
    params = NoWarningParameter(default=default_params)
    T = luigi.FloatParameter(default=getdefault('T'))
    model_dt = luigi.FloatParameter(default=getdefault('dt'))
    datadir = luigi.Parameter(default=defaults.datadir)
    lag_dt = luigi.FloatParameter(default=defaults.lag_dt)

class Trace(Task):
    seed = luigi.IntParameter(0)

    #savelen = luigi.Parameter(default='all')
    #debugprint = luigi.BoolParameter(default=False, significant=False)

    def run(self):
        rndstream = shim.cf.RandomStreams(self.seed)
        model = self.Model(self.params, 'x_linear', T=self.T, dt=self.model_dt, random_stream=rndstream)
        model.x.pad(model.params.Δ.get_value())
        std = model.params.q.get_value()*np.sqrt(model.dt)
        model.x[:model.x.t0idx] = shim.eval(rndstream.normal(std=std, size=(model.x.t0idx,1)))
        #model.x[:2*model.x.t0idx] = 0
        model.advance('end')
        x = anlz.decimate(model.x, target_dt=self.lag_dt)
        with self.output().open('w') as f:
            ml.iotools.save(f, x, format='npr')
        return x

    def output(self):
        cachename = get_cache_name(self.Model, self.datadir, self.T, self.seed, self.params)
        return luigi.local_target.LocalTarget(cachename, format=luigi.format.Nop)

    def load(self):
        return ml.iotools.load(self.output(), format='npr')

def tracetasks(t, seeds,
               Model=DefaultModel,
               params=DefaultModel.Parameters(),
               model_dt=getdefault('dt'),
               datadir=defaults.datadir,
               lag_dt=defaults.lag_dt,
               cores=1):
    get_trace_from_seed = partial(get_trace, T=t, Model=Model,
                                  params=params, model_dt=model_dt,
                                  datadir=datadir, lag_dt=lag_dt)
    if cores == 1:
        for seed in tqdm(seeds, leave=False):
            yield get_trace_from_seed(seed)
    else:
        with Pool(cores) as pool:
            imap = pool.imap_unordered(get_trace_from_seed,
                                       seeds,
                                       chunksize=10) # chunksize is a guess
            for trace in tqdm(imap, total=len(seeds), leave=False):
                yield trace

def get_trace(seed, T,
               Model=DefaultModel,
               params=DefaultModel.Parameters(),
               model_dt=getdefault('dt'),
               datadir=defaults.datadir,
               lag_dt=defaults.lag_dt):
    return Trace(Model=Model, params=params, seed=seed,
                 T=T, model_dt=model_dt, datadir=datadir, lag_dt=lag_dt)
def load_trace(seed, T,
               Model=DefaultModel,
               params=DefaultModel.Parameters(),
               model_dt=getdefault('dt'),
               datadir=defaults.datadir,
               lag_dt=defaults.lag_dt):
    # TODO: Deprecate
    # Don't use nested function (closure) with multiprocessing
    return Trace(Model=Model, params=params, seed=seed,
                 T=T, model_dt=model_dt, datadir=datadir, lag_dt=lag_dt).load()
def traces(t, seeds, cores=1, **kwargs):
    # Run all the tasks to fill the data cache
    # TODO: Find a way to avoid doing this if cache is already full
    luigi.build(tracetasks(t, seeds=seeds, **kwargs))
    # Return the cached calculations
    # With > 1000 cached realizations, we really start to benefit from parallelized I/O
    if cores == 1:
        # Don't use multiprocessing
        # Some reasons to want to deactivate parallelization:
        #   - For debugging
        #   - This call is part of larger task, which is already parallelized
        for trace in tracetasks(t, seeds=seeds, **kwargs):
            yield trace
    else:
        with Pool(cores) as pool:
            load_trace_from_seed = partial(load_trace, T=t, **kwargs)
            imap = pool.imap_unordered(load_trace_from_seed,
                                       seeds,
                                       chunksize=10)
                                       # chunksize is mostly a guess
            tracemap = tqdm(imap, total=len(seeds), leave=False)
            for trace in tracemap:
                yield trace

class TraceIterator:
    """Provides whatever slicing capabilities `seeds` provides."""
    def __init__(self, t, seeds, cores=1, **kwargs):
        self.kwargs = {'t': t, 'cores': cores}
        self.kwargs.update(kwargs)
        self.seeds = seeds
    def __iter__(self):
        return traces(seeds=self.seeds, **self.kwargs)
    def __getitem__(self, key):
        return TraceIterator(seeds=self.seeds[key], **self.kwargs)
class StateIterator:
    """Provides whatever slicing capabilities `seeds` provides."""
    def __init__(self, t, statelen, seeds, cores=1, **kwargs):
        self.kwargs = {'t': t, 'cores': cores}
        self.kwargs.update(kwargs)
        self.seeds = seeds
        self.statelen = statelen
    def __iter__(self):
        for trace in traces(seeds=self.seeds, **self.kwargs):
            yield get_state(trace, t=self.kwargs['t'], statelen=self.statelen)
    def __getitem__(self, key):
        return StateIterator(seeds=self.seeds[key], **self.kwargs)


def get_value(trace, t):
    if isinstance(trace, Trace):
        trace = trace.load()
    elif isinstance(trace, (str, io.IOBase, luigi.LocalTarget)):
        trace = ml.iotools.load(trace, format='npr')
    tidx = trace.get_tidx(t, allow_rounding=True)
    return trace[tidx]
def get_state(trace, t, statelen):
    if isinstance(trace, Trace):
        trace = trace.load()
    elif isinstance(trace, (str, io.IOBase, luigi.LocalTarget)):
        trace = ml.iotools.load(trace, format='npr')
    tidx = trace.get_tidx(t, allow_rounding=True)
    statelen = trace.index_interval(statelen)
    return trace[tidx-statelen:tidx]

def _iter_memoized(memoized_fn, t, params, N, cores, **kwargs):
    if not isinstance(t, Iterable):
        return memoized_fn(t, params, N, **kwargs)
    else:
        if cores == 1:
            _μ = partial(memoized_fn, params=params, N=N, cores=1, **kwargs)
            iterable = map(_μ, t)
            if isinstance(t, np.ndarray):
                return np.fromiter(iterable, np.float, count=len(t))
            else:
                return list(iterable)
        else:
            with Pool(cores) as pool:
                # _μ = partial(memoized_fn, params=params, N=N, cores=1,
                #              pool=pool, **kwargs)
                # async_res = [_μ(s) for s in t]
                async_res = [memoized_fn(t=s, params=params, N=N, cores=1,
                                         pool=pool, **kwargs)
                             for s in t]
                res = [r.get() for r in tqdm(async_res)]
                if isinstance(t, np.ndarray):
                    return np.array(res)
                else:
                    return res

# Using lru_cache along with multiprocessing brings about special challenges
# Part if this is solved by specializing lru_cache to detect the 'pool'
# argument and dispatch jobs to it.
# Thus this approach requires that the function dispatched to the mp pool be
# picklable; however, because @lru_cache hides the function it decorates, it
# can't be sent to pool.
# So the actual calculations (e.g. μ, Σ) need to be in a function which is not
# overwritten, which we provide as _μ and _Σ.

def _μ(t, params, N, **kwargs):
    seeds = N if isinstance(N, Iterable) else list(range(N))
    return sum(get_value(trace, t)
               for trace in traces(t, params=params, seeds=seeds, **kwargs)) / len(seeds)
@lru_cache(inner=_μ)
def _memoized_μ(t, params, N, **kwargs):
    pass
def μ(t, params, N, cores=1, **kwargs):
    return _iter_memoized(_memoized_μ, t, params, N, cores=cores, **kwargs)

@lru_cache()
def stateμ(t, params, N, statelen, cores=1, **kwargs):
    seeds = N if isinstance(N, Iterable) else list(range(N))
    return sum(get_state(trace, t, statelen)
               for trace in traces(t, params=params, seeds=seeds, **kwargs)) / len(seeds)
def _Σ(t, params, N, **kwargs):
    seeds = N if isinstance(N, Iterable) else list(range(N))
    xgen = (get_value(trace, t)
            for trace in traces(t, params=params, seeds=seeds, **kwargs))
    mu = μ(t, params, N, **kwargs)
    return sum(x**2 for x in xgen) / len(seeds) - mu**2
@lru_cache(inner=_Σ)
def _memoized_Σ(t, params, N, **kwargs):
    pass
def Σ(t, params, N, cores=1, **kwargs):
    return _iter_memoized(_memoized_Σ, t, params, N, cores=cores, **kwargs)

@lru_cache()
def stateΣ(t, params, N, statelen, cores=1, **kwargs):
    seeds = N if isinstance(N, Iterable) else list(range(N))
    states = (get_state(trace, t, statelen)
              for trace in traces(t, params=params, seeds=seeds, **kwargs))
    _μ = μ(t, params, N, statelen, **kwargs)
    return sum(np.multiply((state-_μ).T, state-_μ)
               for state in states) / len(seeds)

@lru_cache()
def PCA(t, params, N, statelen, cores=1, **kwargs):
    return np.linalg.eig(stateΣ(t, params, N, statelen, cores=1, **kwargs))

class Realizations:
    """
    This class is used to representa an ensemble of realizations.
    It provides memoized statistical methods (like μ and Σ): their initial execution
    is slower when we have more realizations/seeds, but subsequent calls just
    returned the cached value
    Each realization is stored on disk in `datadir` (default: 'data'). So the
    very first time this run, it will take a while to generate all the realizations.
    Subsequent calls to the same realizations require only reading from disk,
    which is much faster. The Luigi package is used to manage this disk cache.
    The Realizations class is careful never to actually load all realizations in
    memory at once, so statistics over large numbers of realizations (1,000 or more)
    are possible. The drawback with this approach is that the same realization
    may be loaded from disk multiple times. To minimize the I/O bottleneck,
    we use Python's `multiprocessing` module to load from the disk cache.
    The number of CPU cores used for I/O is set by the `cores` parameter.
    """

    def __init__(self, Model, params, seeds, statelen, T,
                 cores=defaults.cores,
                 model_dt=getdefault('dt'),
                 datadir=defaults.datadir,
                 lag_dt=defaults.lag_dt,
                ):
        """

        Parameters
        ----------
        Model: Class
            One of the models in models.py. Note that we want the class, not
            an instance.
        params: Parameter object
            Instance of Model.Parameters.
        seeds: list of ints
            The seeds identifying the different realizations.
            # FIXME: Actually only the length of seeds is used
        statelen: float
            The length of a state of the dynamical system, i.e. it's longest
            delay.
        T: float
            Length of realizations
        cores: int
            Number of cores over which to parallelize data loading operations.
            Note that this does /not/ affect the number of workers attached to
            the Luigi scheduler.
            Using multiple cores helps compensate the substantial I/O overhead
            of using an on-disk cache for realizations.
        """
        self.Model = Model
        self.params = params
        self.seeds = tuple(seeds)  # Make list of seeds hashable
        self.statelen = float(statelen)  # Make sure it's not an int
        self.T = float(T)  # Make sure it's not an int
        self.cores = cores
        self.model_dt = model_dt  # TODO: Use the dt from a reference instead ?
        self.datadir = datadir
        self.lag_dt = lag_dt   # FIXME: This isn't actually used (the global default is)

        self.reftrace = next(iter(self.traces()[0:1]))
        # Traces are saved already decimated, i.e. with lag_dt instead of model_dt

    def __hash__(self):
        return hash((self.Model, self.params, self.seeds, self.statelen,
                     self.T, self.model_dt, self.lag_dt))

    @property
    def kwargs(self):
        return {'Model': self.Model,
                'params': self.params,
                'model_dt': self.model_dt,
                'datadir': self.datadir,
                'lag_dt': self.lag_dt}

    def set(self):
        """'Set' the realizations by computing all their traces.
        Realizations are computed up to their latest possible time as set by
        `self.T`.
        It's usually a good idea to call this function before others, to ensure
        calculations are not done twice (e.g. by asking first for μ(1), then
        for μ(2)).
        """
        luigi.build(tracetasks(self.T, seeds=self.seeds, cores=self.cores, **self.kwargs))

    @property
    def state_axis(self):
        nstops = self.reftrace.index_interval(self.statelen)
        return anlz.Axis(label='τ',
                         stops=np.arange(nstops)*self.lag_dt)

    def traces(self, t=None, seeds=None):
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return TraceIterator(t, seeds=seeds, cores=self.cores, **self.kwargs)
    def states(self, t=None, seeds=None):
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return StateIterator(t, statelen=self.statelen, seeds=seeds,
                             cores=self.cores, **self.kwargs)

    def μ(self, t=None, seeds=None):
        """Pointwise mean"""
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return μ(t, N=len(seeds), cores=self.cores, **self.kwargs)
    def stateμ(self, t=None, seeds=None):
        """State mean"""
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return stateμ(t, N=len(seeds), statelen=self.statelen, cores=self.cores, **self.kwargs)
    def Σ(self, t=None, seeds=None):
        """Pointwise variance"""
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return Σ(t, N=len(seeds), cores=self.cores, **self.kwargs)
    def stateΣ(self, t=None, seeds=None):
        """State variance, i.e. autocovariance."""
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return stateΣ(t, N=len(seeds), statelen=self.statelen, cores=self.cores, **self.kwargs)
    def σ(self, t=None, seeds=None):
        """Standard deviation"""
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        return np.sqrt(self.Σ(t))
    def crossΣ(self, N, t=None):
        """
        Return the cross-covariance across realizations.

        Parameter
        ---------
        N: int
            Number of realizations to use; they will be picked at random.
            Note that a total of N² realizations will be used: N are chosen and
            for each, we compute the cross-covariance with N other realizations.
            We then take the sample average to get the cross-covariance of the
            ensemble.
        t: float (optional)
            Time at which to take states. If unspecified, the latest time is
            used (as set by `self.T`).

        ..Note: At the moment this function is not parallelized.
        """
        if t is None: t=self.T
        if N > np.sqrt(len(self.seeds)):
            logger.warning("Number of realizations used to compute cross-covariance "
                           "shouldn't exceed sqrt(# realizations). \n"
                           "(# realizations: {}.)".format(len(self.seeds)))
        seeds1 = np.random.choice(self.seeds, N, replace=False)
            # We use replace=False because N may be relatively small (e.g. 20)
            # Accidentaly reusing the same sample then can strongly skey the result
        seeds2pool = [seed for seed in self.seeds if seed not in seeds1]
            # All seeds that aren't one of the base seeds
        def get_seeds2():
            return np.random.choice(seeds2pool, N, replace=False)
        μ = self.stateμ()

        # if self.cores == 1:
        if True:
            return sum(np.multiply((state1-μ).T, state2-μ)
                       for state1 in tqdm(self.states(seeds=seeds1), total=N)
                       for state2 in self.states(seeds=get_seeds2())
                       ) / N / (N-1)
        # else:
        #     with Pool(self.cores) as pool:
        #         imap = pool.imap_unordered()

    def PCA(self, t=None, seeds=None):
        """Eigenvalue decomposition of autocovariance matrix."""
        if t is None: t=self.T
        if seeds is None: seeds=self.seeds
        pca = PCA(t, N=len(seeds), statelen=self.statelen,
                  cores=self.cores, **self.kwargs)
        return (np.real_if_close(pca[0]), np.real_if_close(pca[1]))
    def PCA_w(self, t=None, seeds=None):
        """PCA component coefficients."""
        return self.PCA(t, seeds)[0]
    def PCA_φ(self, t=None, seeds=None):
        """PCA component vectors."""
        return self.PCA(t, seeds)[1]


#====================================
# Luigi cache management
#====================================

def get_cache_name(Model, datadir, minT, seed, params):
    """
    Return the cache name which matches 'params' and 'seed' and for which
    'T' is at least 'minT'.
    """

    Cache = namedtuple('Cache', ['filename', 'params'])
    CacheRef = namedtuple('CacheRef', ['index', 'cache'])
    def getpvals(f):
        #if legacy:
        #    pvalstrs = f[4:].split('__')    # Parameters are separated by '__'
        #else:
        # Remove any extension
        ext = os.path.splitext(f)[1]
        if len(ext) > 0 and ext[1:].isalpha():
            f = os.path.splitext(f)[0]
        pvalstrs = f.split('__')
        prefix, pvalstrs = pvalstrs[0], pvalstrs[1:]
        def split_val(s):
            # Separates the leading parameter name from the value
            for i, c in enumerate(s):
                if c.isdigit():
                    break
            if i > len(s):
                raise RuntimeError("Misformed cache filename: '{}'".format(f))
            val = s[i] + s[i+1:].replace('-', '.')  # Don't replace '-' if it indicates negative number
            if val.count('.') <= 1 and val.replace('.', '').replace('-', '').isdigit():
                # Make sure value is a proper number before attempting conversion
                val = float(val) if '.' in val else int(val)
            return s[:i], val
        return OrderedDict(split_val(s) for s in pvalstrs)

    if len(datadir) == 0:
        self.datadir = os.curdir

    # Ensure the cache directory exists
    os.makedirs(datadir, exist_ok=True)

    paramstr = '__seed{}__{}'.format(seed, params_to_string(params))
    cacheprefix = Model.__name__
    Δc = len(cacheprefix)
    caches = [Cache(f, getpvals(f))
              for f in os.listdir(datadir)
              if f[:Δc] == cacheprefix and paramstr in f]
    #joinstr = '_' if legacy else '__'
    joinstr = '__'
    cachename =  cacheprefix + joinstr + 'T' + str(minT).replace('.', '-') + paramstr
        # If we can't find a cache, this is the name to use to create a new one

    if len(caches) > 0:
        # Try to find a suitable pre-existing cache
        maxcache = CacheRef(0, caches[0])
        for i, cache in enumerate(caches[1:], start=1):
            if cache.params['T'] > maxcache.cache.params['T']:
                # TODO: Mark unkept cache for deletion
                maxcache = CacheRef(i, cache)
        if maxcache.cache.params['T'] < minT:
            #return cachename
            pass
        else:
            cachename = maxcache.cache.filename
    return os.path.join(datadir, cachename)

def params_to_string(params):
    joinstr = '__'
    if isinstance(params, tuple) and hasattr(params, '_fields'):
        # namedtuple
        return joinstr.join(s + str(getattr(params, s)) for s in params._fields).replace('.', '-')
    else:
        sortednames = sorted(params.keys())
        return joinstr.join(name + str(params[name]) for name in sortednames).replace('.', '-')

#===========================================

def cast_params(params):
    if params is None:
        params = globals()['params']
    return params
