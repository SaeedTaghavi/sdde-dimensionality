from init import *
import io
from models import TanhModel as DefaultModel

defaults = {
    'seeds': list(range(10)),
    'datadir': 'test-cache',
    'lag_dt': 0.01
}

default_params = DefaultModel.Parameters()
class Task(luigi.Task):
    Model = luigi.Parameter(default=DefaultModel)
    params = luigi.Parameter(default=default_params)
    T = luigi.FloatParameter(default=2*default_params.Δ)
    model_dt = luigi.FloatParameter(default=DefaultModel.defaults['dt'])
    datadir = luigi.Parameter(default=defaults['datadir'])
    lag_dt = luigi.FloatParameter(default=defaults['lag_dt'])

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
            print("\n\n" + f.name + "\n\n")
            ml.iotools.save(f, x, format='npr')
        return x

    def output(self):
        #import pdb; pdb.set_trace()
        cachename = get_cache_name(self.Model, self.datadir, self.T, self.seed, self.params)
        return luigi.local_target.LocalTarget(cachename, format=luigi.format.Nop)

    def load(self):
        return ml.iotools.load(self.output(), format='npr')

def tracetasks(t, seeds,
               Model=DefaultModel,
               params=DefaultModel.Parameters(),
               model_dt=DefaultModel.defaults['dt'],
               datadir=defaults['datadir'],
               lag_dt=defaults['lag_dt']):
    for seed in seeds:
        yield Trace(Model=Model, params=params, seed=seed,
                           T=t, model_dt=model_dt, datadir=datadir,
                           lag_dt=lag_dt)

def traces(*args, **kwargs):
    luigi.build(tracetasks(*args, **kwargs))
    for task in tracetasks(*args, **kwargs):
        yield task.load()

def get_state(trace, t, statelen):
    if isinstance(trace, (str, io.IOBase, luigi.LocalTarget)):
        trace = ml.iotools.load(trace, format='npr')
    tidx = trace.get_tidx(t)
    statelen = trace.index_interval(statelen)
    return trace[tidx-statelen:tidx]

@lru_cache()
def μ(t, params, N, statelen, **kwargs):
    seeds = list(range(N))
    return sum(get_state(trace, t, statelen)
               for trace in traces(t, params=params, seeds=seeds, **kwargs)) / len(seeds)
@lru_cache()
def Σ(t, params, N, statelen, **kwargs):
    seeds = list(range(N))
    states = (get_state(trace, t, statelen)
              for trace in traces(t, params=params, seeds=seeds, **kwargs))
    _μ = μ(t, params, N, statelen, **kwargs)
    return sum(np.multiply((state-_μ).T, state-_μ)
               for state in states) / len(seeds)

class Realizations:

    def __init__(self, Model, params, seeds, statelen, t,
                 model_dt=DefaultModel.defaults['dt'],
                 datadir=defaults['datadir'],
                 lag_dt=defaults['lag_dt']):
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
        t: float
            The time at which to take the state (i.e. how long to simulate for)
        """
        self.Model = Model
        self.params = params
        self.seeds = seeds
        self.statelen = float(statelen)  # Make sure it's not an int
        self.t = float(t)  # Make sure it's not an int
        self.model_dt = model_dt
        self.datadir = datadir
        self.lag_dt = lag_dt

    @property
    def kwargs(self):
        return {'Model': self.Model,
                'model_dt': self.model_dt,
                'datadir': self.datadir,
                'lag_dt': self.lag_dt}

    @property
    def traces(self):
        return traces(self.t, seeds=self.seeds, **self.kwargs)

    @property
    def μ(self):
        return μ(self.t, self.params, len(self.seeds), self.statelen, **self.kwargs)

    @property
    def Σ(self):
        return Σ(self.t, self.params, len(self.seeds), self.statelen, **self.kwargs)

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
