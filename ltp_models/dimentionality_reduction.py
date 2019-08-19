import argparse

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler

# idx = header.index("Ca")
# d[:, idx]
from ltp_models.utils import get_data

COMPONENT = 0

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--morphology", nargs='+', required=True)

    ap.add_argument("--method", help="pca or nnmf. default=nmf", default='nmf')
    ap.add_argument("--trial_agregation", help="avg or concat. default=concat", default='concat')

    args = ap.parse_args()

    for m in args.morphology:
        header, data = get_data(folder=args.folder, prefix=args.prefix, morpho=m, no_time=True)

        if args.trial_agregation == "concat":
            data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
        elif args.trial_agregation == "avg":
            data = np.average(data, axis=2)
        else:
            raise NameError("Allowed trial agregation are: 'avg', 'concat'.")

        data = MinMaxScaler().fit_transform(data)

        f = PCA(n_components=args.component_number)
        c = f.fit_transform(data)
        print('PCA variance explained:', m, sum(f.explained_variance_ratio_), f.explained_variance_ratio_)

        importance_pca = sorted(
            list(zip(header, np.average(np.dot(c[:, COMPONENT].reshape(len(c), 1), f.components_[COMPONENT, :].reshape(f.components_.shape[1], 1).T),
                                        axis=1).tolist())), key=lambda x: -x[1])

        f = NMF(n_components=args.component_number, init='random', random_state=33)
        c = f.fit_transform(data)
        print('NMF variance explained:', m, np.sum(c), sum(c))

        importance_nmf = sorted(list(zip(header, c[:, COMPONENT])), key=lambda x: -x[1])

        print('%s COMPONENT IMPORTANCE FOR:' % (COMPONENT+1), 'PCA', 'NMF')
        for i in zip(importance_pca, importance_nmf):
            print(i)
