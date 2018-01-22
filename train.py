from chainer import optimizers, training, iterators, datasets
from chainer.training import extensions
from nvil import SBN


def main():
    train_x, _ = datasets.get_mnist(withlabel=False)
    train_x = train_x.astype('f')
    train_iter = iterators.SerialIterator(
        train_x, batch_size=128, shuffle=True)

    dims = [200, 200]
    x_dim = train_x.shape[1]
    model = SBN(dims, x_dim)
    optimizer = optimizers.Adam()
    optimizer.setup(model)
    updater = training.StandardUpdater(train_iter, optimizer, device=0)
    trainer = training.Trainer(updater, (100, 'epoch'))
    trainer.extend(extensions.LogReport())
    trainer.extend(
        extensions.PrintReport(
            [
                'epoch', 'main/loss', 'main/-logp', 'main/-logq',
                'main/var',
            ]))
    trainer.extend(
        extensions.snapshot_object(
            model, filename='snapshot_epoch_{.updater.epoch:03d}.npz'),
        trigger=(1, 'epoch'))
    trainer.run()


if __name__ == '__main__':
    main()
