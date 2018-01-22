import os
import re
import matplotlib
from chainer import datasets, serializers
from nvil import SBN
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa
import seaborn as sns  # noqa


def main():
    _, test_x = datasets.get_mnist(withlabel=False)
    test_x = test_x.astype('f')[:25]

    for i in range(25):
        plt.subplot(5, 5, i+1)
        plt.imshow(test_x[i].reshape(28, 28), cmap='gray_r')
    plt.savefig('./result/ans.png')
    plt.close()

    x_dim = test_x.shape[1]
    model = SBN([200, 200], x_dim)
    pattern = re.compile(r'.+npz$')
    for fname in sorted(os.listdir('./result')):
        if not pattern.match(fname):
            continue
        out = './result/{}.png'.format(fname)
        if os.path.exists(out):
            continue

        print(fname)
        serializers.load_npz(os.path.join('./result', fname), model)

        gen_x = model.generate(test_x)
        gen_x = gen_x.reshape(-1, 28, 28)
        for i in range(25):
            plt.subplot(5, 5, i+1)
            plt.imshow(gen_x[i], cmap='gray_r')
        plt.savefig(out)
        plt.close()


if __name__ == '__main__':
    main()
