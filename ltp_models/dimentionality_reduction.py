import argparse

import numpy as np
from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler

# idx = header.index("Ca")
# d[:, idx]
from ltp_models.utils import get_data


def prepare_trails(data, agregation):
    """
    :param data:
    :param agregation:
        avg or concat
    :return:
    """
    if agregation == "concat":
        data = data.reshape([data.shape[0], data.shape[1] * data.shape[2]])
    elif agregation == "avg":
        data = np.average(data, axis=2)
    else:
        raise NameError("Allowed trial agregation are: 'avg', 'concat'.")

    return data


def pca(data, header, comp_num, importance_component_num=1):
    ic = importance_component_num - 1
    f = PCA(n_components=comp_num)
    c = f.fit_transform(data)

    a = c[:, 0].reshape(len(c), 1)
    b = f.components_[ic, :].reshape(f.components_.shape[1], 1)
    avg_importance = np.average(np.dot(a, b.T), axis=1).tolist()

    return f, c, list(zip(header, avg_importance))


def nmf(data, header, comp_num, importance_component_num=1, init='random', random_state=33):
    ic = importance_component_num - 1
    f = NMF(n_components=comp_num, init=init, random_state=random_state)
    c = f.fit_transform(data)

    avg_importance = c[:, ic]

    return f, c, list(zip(header, avg_importance))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--morphology", nargs='+', required=True)
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--agregation", help="Many trial agregation type: avg, concat. default=concat", default='concat')

    args = ap.parse_args()

    for m in args.morphology:
        header, data = get_data(folder=args.folder, prefix=args.prefix, morpho=m, no_time=True)

        data = prepare_trails(data, agregation=args.agregation)
        data = MinMaxScaler().fit_transform(data)

        pca_f, pca_c, pca_i = pca(header=header, data=data, comp_num=args.component_number)
        nmf_f, nmf_c, nmf_i = nmf(header=header, data=data, comp_num=args.component_number)

        print('PCA variance explained:', round(sum(pca_f.explained_variance_ratio_)*100, 5))
        print('NMF variance explained:', round(np.sum(nmf_c), 4))
        print()

        pca_i = sorted(pca_i, key=lambda x: -x[1])
        nmf_i = sorted(nmf_i, key=lambda x: -x[1])

        print("PCA / NMF")
        for i in list(zip(pca_i, nmf_i))[:10]:
            pcaval = "%s:%s" % (i[0][0], round(i[0][1], 4))
            nmfval = "%s:%s" % (i[1][0], round(i[1][1], 4))
            print("%s\t%s" % (pcaval, nmfval))
