from init import *

class Model(sinn.models.Model):
    Parameter_info = OrderedDict( (('α', ('floatX', -1.)),
                                   ('β', ('floatX', 1.)),
                                   ('Δ', ('floatX', 1.)),
                                   ('q', ('floatX', 1.))
                                  ) )
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['x'])

    def __init__(self, params, xname='x', T=10, dt=0.001, random_stream=None):
        if isinstance(params, dict):
            # Required because Model still uses old custom Parameters class
            params = self.Parameters(**params)
        self.x = Series(name=xname, time_array=np.arange(0, T, dt), shape=(1,), dtype=np.float64)
        super().__init__(params, reference_history=self.x)

        self.rndstream = random_stream
        self.x.set_update_function(self.x_fn)
        self.x.add_input(self.x)
        self.add_history(self.x)

        self.statehists = [ getattr(self, varname) for varname in self.State._fields ]

    # Using a static method allows us to call F without a class instance, e.g.
    # `TanhModel.F(α,β)` to draw the force function.
    @staticmethod
    def F(x, α, β):
        # Must be implemented in subclasses
        raise NotImplementedError

    def x_fn(self, t):
        Δi = self.x.index_interval(self.params.Δ)
        tidx = self.x.get_t_idx(t)
        if shim.isscalar(t):
            outshape = self.x.shape
        else:
            tshape = shim.asarray(t).shape
            outshape = shim.concatenate(tshape, shim.asarray(self.x.shape))
        return (self.x[tidx-1] + self.F(self.x[tidx-Δi], self.params.α, self.params.β)*self.x.dt
                + self.params.q*self.rndstream.normal(size=outshape, avg=0, std=shim.sqrt(self.x.dt)))

    def symbolic_update(self, tidx, xΔ, x0):
        #curstate = self.State(*statevars)
        outshape = self.x.shape

        xt = (x0 + self.F(xΔ, self.params.α, self.params.β)*self.x.dt
              + self.params.q*self.rndstream.normal(size=outshape, avg=0, std=shim.sqrt(self.x.dt)))

        return [xt], {}
        #return [], {}

class TanhModel(Model):
    @staticmethod
    def F(x, α, β):
        return α*(2*β) * shim.tanh(x/(2*β))

class LinearModel(Model):
    @staticmethod
    def F(x, α, β):
        return α * x
