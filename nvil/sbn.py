import math
import chainer
from chainer import (
    Chain, report, ChainList
)
from chainer import links as L
from chainer import functions as F


class SBNInferenceModel(ChainList):

    def __init__(self, dims):
        super(SBNInferenceModel, self).__init__()
        with self.init_scope():
            for dim in dims:
                self.add_link(L.Linear(None, out_size=dim))

    def sample_latent(self, x):
        '''
        h ~ Q(h|x)
        '''
        hs = []
        xp = chainer.cuda.get_array_module(x)
        prev = x
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            for link in self.links(skipself=True):
                mu = F.sigmoid(link(prev))
                mask = xp.random.random(mu.shape)
                hs.append((mu.data > mask).astype('f'))
                prev = mu

        return hs

    def nll(self, x, hs):
        '''
        -logQ(h|x)
        '''
        nll = []
        prev = x
        for link, h in zip(self.links(skipself=True), hs):
            mu = link(prev)
            nll.append(F.sum(F.bernoulli_nll(h, mu, 'no'), axis=1))
            prev = h

        return nll


class SBNGenerativeModel(ChainList):

    def __init__(self, dims, x_dim):
        super(SBNGenerativeModel, self).__init__()
        with self.init_scope():
            self.bias = chainer.Parameter(0, shape=(dims[-1]))
            self.add_link(L.Linear(None, out_size=x_dim))
            for dim in dims[:-1]:
                self.add_link(L.Linear(None, out_size=dim))

    def nll(self, x, hs):
        '''
        -logP(x|h), -logP(h)
        '''
        nll_pxh = []
        prev = x
        # P(x|h)
        for link, h in zip(self.links(skipself=True), hs):
            mu = link(h)
            nll_pxh.append(F.sum(F.bernoulli_nll(prev, mu, 'no'), axis=1))
            prev = h

        # P(h)
        batchsize = hs[-1].shape[0]
        nll_ph = F.sum(
            F.bernoulli_nll(
                hs[-1],
                F.broadcast_to(
                    self.bias, (batchsize, self.bias.shape[0])
                ),
                'no',
            ),
            axis=1,
        )

        return nll_pxh, nll_ph

    def generate(self, h):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            # P(x|h)
            for link in reversed(list(self.links(skipself=True))):
                mu = F.sigmoid(link(h))
                h = mu

            return h.data


class Baseline(Chain):

    def __init__(self):
        super(Baseline, self).__init__()
        with self.init_scope():
            self.l0 = L.Linear(None, 200)
            self.l1 = L.Linear(None, 1)

    def __call__(self, x):
        h0 = F.tanh(self.l0(x))
        return self.l1(h0).reshape(-1)


class Baselines(ChainList):

    def __init__(self, n):
        super(Baselines, self).__init__()
        with self.init_scope():
            for i in range(n):
                self.add_link(Baseline())

    def __call__(self, x):
        return [b(x) for b in self.links(skipself=True)]


class SBN(Chain):

    def __init__(self, dims, x_dim):
        super(SBN, self).__init__()
        self.alpha = 0.9
        self.mean = self.xp.zeros(len(dims))
        self.var = self.xp.zeros(len(dims))
        with self.init_scope():
            self.inf_model = SBNInferenceModel(dims)
            self.gen_model = SBNGenerativeModel(dims, x_dim)
            self.baselines = Baselines(len(dims))

    def __call__(self, x):
        xp = self.xp

        # sample latent variable
        h = self.inf_model.sample_latent(x)

        # compute learning signal
        nll_qhx = self.inf_model.nll(x, h)
        nll_phx, nll_ph = self.gen_model.nll(x, h)
        bs = [b(x) for b in self.baselines]
        signals = [(p-q-b).data for p, q, b in zip(nll_phx, nll_qhx, bs)]

        # update statistics
        new_mean = [xp.mean(s) for s in signals]
        new_var = [xp.sum(xp.power(s-m, 2)) for s, m in zip(signals, new_mean)]
        self.mean = [self.alpha*m + (1.-self.alpha)*nm for m, nm in zip(
            self.mean, new_mean)]
        self.var = [self.alpha*v + (1.-self.alpha)*nv for v, nv in zip(
            self.var, new_var)]

        signals = [s-m for s, m in zip(signals, self.mean)]
        signals = [s/max(1, math.sqrt(v)) for s, v in zip(signals, self.var)]

        loss = 0
        loss += F.mean(sum(nll_phx)+nll_ph)
        loss += 0.2*F.mean(sum([-s*q for s, q in zip(signals, nll_qhx)]))
        loss += F.mean(sum([-s*b for s, b in zip(signals, bs)]))

        report({
            'loss': loss, '-logp': F.mean(sum(nll_phx))+F.mean(nll_ph),
            '-logq': F.mean(sum(nll_qhx)), 'var': sum(self.var)
        }, self)

        return loss

    def generate(self, x):
        with chainer.using_config('train', False), chainer.no_backprop_mode():
            h = self.inf_model.sample_latent(x)[-1]
            gen_x = self.gen_model.generate(h)

        return gen_x
