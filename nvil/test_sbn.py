from nvil.sbn import SBNInferenceModel, SBNGenerativeModel
import cupy as cp


def _sample(shape):
    return (cp.random.randn(*shape) > cp.random.random(shape)).astype('f')


def test_inference_model():
    cp.random.seed(215)
    dims = [200, 100]
    batchsize = 10
    inf_model = SBNInferenceModel(dims)
    inf_model.to_gpu()
    x = _sample((batchsize, 25))
    hs = inf_model.sample_latent(x)
    assert len(hs) == len(dims)
    assert hs[0].shape == (batchsize, dims[0])
    assert hs[1].shape == (batchsize, dims[1])

    nll = inf_model.nll(x, hs)
    assert len(nll) == len(dims)
    assert nll[0].shape == (batchsize,)
    assert nll[1].shape == (batchsize,)


def test_generative_model():
    cp.random.seed(215)
    dims = [200, 100]
    x_dim = 25
    batchsize = 10
    gen_model = SBNGenerativeModel(dims, x_dim)
    gen_model.to_gpu()
    x = _sample((batchsize, x_dim))
    hs = [
        _sample((batchsize, dims[0])),
        _sample((batchsize, dims[1])),
    ]

    nll_pxh, nll_ph = gen_model.nll(x, hs)
    assert len(nll_pxh) == len(dims)
    assert nll_pxh[0].shape == (batchsize,)
    assert nll_pxh[1].shape == (batchsize,)
    assert nll_ph.shape == (batchsize,)
