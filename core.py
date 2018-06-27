from init import *
import io
from models import TanhModel as DefaultModel

defaults = {
    'seeds': list(range(10)),
    'datadir': 'test-cache',
    'lag_dt': 0.01
}

class Task(luigi.Task):
    default_params = DefaultModel.Parameters()
    Model = luigi.Parameter(default=DefaultModel)
    params = luigi.Parameter(default=default_params)
    T = luigi.FloatParameter(default=2*default_params.Δ)
    model_dt = luigi.FloatParameter(default=DefaultModel.defaults['dt'])
    datadir = luigi.Parameter(default=defaults['datadir'])
    lag_dt = luigi.FloatParameter(default=defaults['lag_dt'])

class InMemoryTarget(luigi.Target):
    """
    Target that is stored only in program memory. Since storage is not
    permanent, this should only be used either for cheap operations (e.g.
    at the final analysis stage) or for intermediate steps which don't need
    to be saved to disk.
    """
    memory = {}  # Class attribute, shared across instances
    def __init__(self, varname):
        """
        `varname`: Variable name or key.
        """
        self.varname = varname
    def exists(self):
        return self.varname in self.memory
    def set(self, value):
        import pdb; pdb.set_trace()
        self.memory[self.varname] = value
    def get(self):
        return self.memory[self.varname]

class Calc(luigi.Task):
    def run(self):
        self.output().set(self.calc())
    def output(self):
        return InMemoryTarget(hash(self))

def calc(task, generator=False):
    if not isinstance(task, Iterable):
        print("\n\n")
        print(InMemoryTarget.memory)
        luigi.build([task])
        print(InMemoryTarget.memory)
        print("\n\n")
        return task.output().get()
    else:
        luigi.build(task)
        gen = (t.output.get() for t in task)
        if generator:
            return gen
        else:
            return tuple(gen)

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
            ml.iotools.save(f, x)
        return x

    def output(self):
        #import pdb; pdb.set_trace()
        cachename = get_cache_name(self.Model, self.datadir, self.T, self.seed, self.params)
        return luigi.local_target.LocalTarget(cachename, format=luigi.format.Nop)

def get_state(trace, t, statelen):
    if isinstance(trace, (str, io.IOBase, luigi.LocalTarget)):
        trace = ml.iotools.load(trace, format='npr')
    tidx = trace.get_tidx(t)
    statelen = trace.index_interval(statelen)
    return trace[tidx-statelen:tidx]

class M(Task, Calc):
    seeds = luigi.ListParameter(defaults['seeds'])
    t = luigi.FloatParameter()

    def requires(self):
        return (Trace(Model=self.Model, params=self.params, seed=seed,
                      T=self.t, model_dt=self.model_dt, datadir=self.datadir,
                      lag_dt=self.lag_dt)
                for seed in self.seeds)

    def calc(self):
        return sum(get_state(trace, self.t, self.params.Δ)
                   for trace in self.input()) / len(self.seeds)

class Cov(Task, Calc):
    seeds = luigi.ListParameter(defaults['seeds'])
    t = luigi.FloatParameter()

    def requires(self):
        # returns (traces, μ)
        return  ( (Trace(Model=self.Model, params=self.params, seed=seed,
                               T=self.t, model_dt=self.model_dt,
                               datadir=self.datadir,
                               lag_dt=self.lag_dt)
                         for seed in self.seeds),
                 #M(**dict(self.get_params())) )
                 M(Model=self.Model, params=self.params, seeds=self.seeds,
                    t=self.t, model_dt=self.model_dt,
                    datadir=self.datadir,
                    lag_dt=self.lag_dt) )

    def calc(self):
        traces, μ = self.input()
        states = (get_state(trace, self.t, self.params.Δ) for trace in traces)
        μ = μ.get()
        return sum(np.multiply((state-μ).T, state-μ)
                   for state in states) / len(self.seeds)

def μ(t, *args, **kwargs):
    return calc(M(t=t, *args, **kwargs))
def Σ(t, *args, **kwargs):
    return calc(Cov(t=t, *args, **kwargs))

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
