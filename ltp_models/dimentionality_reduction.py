import argparse
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA, NMF
from sklearn.preprocessing import MinMaxScaler
from ltp_models.utils import get_data


def plot(name, molecules, data, header):
    c = get_concentration(data, header, molecules=molecules, norm=True, sum_cols=True)
    plt.plot(c, label=name)


def reduce_and_plot(reduction_type, n_components, data):
    reduction_type = reduction_type.lower().strip()
    if reduction_type == 'pca':
        r = PCA
    elif reduction_type == 'nmf':
        r = NMF
    else:
        raise TypeError("Wrong type of reduction class. Allowed: PCA, NMF.")

    data = MinMaxScaler().fit_transform(data)

    f = r(n_components=n_components)
    c = f.fit_transform(data)
    c = MinMaxScaler().fit_transform(c)

    for i in range(0, c.shape[1]):
        plt.plot(c[:, i], label='%s_%s' % (reduction_type, i))

    return f, c


def exclude(data, header, molecules):
    """
    :param data:
    :param header:
    :param molecules:
    :return:
        tuple(data, header) after removal
    """
    idxs = [header.index(m) for m in molecules]

    data = np.delete(data, idxs, axis=1)
    header = np.delete(header, idxs).tolist()

    return data, header


def get_concentration(data, header, molecules, sum_cols=False, norm=False):
    if isinstance(molecules, str):
        molecules = molecules.split(' ')

    v = np.array([data[:, header.index(m)] for m in molecules]).T

    if sum_cols:
        v = np.array([np.sum(v, axis=1)]).T
    if norm:
        v = MinMaxScaler().fit_transform(v)
    if v.shape[0] == 1:
        v = v[0]
    return v


def prepare_trails(data, agregation):
    """
    :param data:
    :param agregation:
        avg or concat
    :return:
    """
    if agregation == "concat":
        data = data.reshape(data.shape[0]*data.shape[1], data.shape[2])
    elif agregation == "avg":
        data = np.average(data, axis=0)
    else:
        raise NameError("Allowed trial agregation are: 'avg', 'concat'.")

    return data


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--trials", nargs='+', help="Trial numbers if required. Default: take all trials", default=None, type=int)
    ap.add_argument("--agregation", help="Many trial agregation type: avg, concat. Default: concat", default='concat')

    args = ap.parse_args()

    data, header = get_data(folder=args.folder, prefix=args.prefix, trials=args.trials, morpho=args.morphology)
    data = prepare_trails(data, agregation=args.agregation)

    to_exclude = ['time', 'Ca', 'Leak']
    to_exclude.extend([h for h in header if "out" in h.lower()])
    to_exclude.extend([h for h in header if "buf" in h.lower()])

    data, header = exclude(data, header, molecules=to_exclude)

    print(header)
    data = data[:1000, :]  # compute only for 1000 steps

    plt.figure(1)
    f, c = reduce_and_plot('nmf', n_components=3, data=data)
    plot("CK", molecules="CK", data=data, header=header)
    plot("Gi", molecules="Gi", data=data, header=header)
    plot("Gibg", molecules="Gibg", data=data, header=header)
    plot("PKA", molecules="PKA", data=data, header=header)
    plot("CKp", molecules="CKp CKpCaMCa4", data=data, header=header)
    plot("CaMCa", molecules="CaMCa2 CaMCa4", data=data, header=header)
    plot("pbAR", molecules="pbAR ppbAR pppbAR ppppbAR", data=data, header=header)
    plt.legend(loc='best')

    plt.figure(2)
    molecules_by_components = MinMaxScaler().fit_transform(f.components_.T)
    x = np.arange(len(header))
    plt.xticks(x, header, rotation=90)
    plt.plot(x, molecules_by_components[:, 0])
    plt.plot(x, molecules_by_components[:, 1])
    plt.plot(x, molecules_by_components[:, 2])

    c1_sorted = sorted(zip(header, molecules_by_components[:, 0]), key=lambda x: -x[1])
    c2_sorted = sorted(zip(header, molecules_by_components[:, 1]), key=lambda x: -x[1])
    c3_sorted = sorted(zip(header, molecules_by_components[:, 2]), key=lambda x: -x[1])

    print(c1_sorted[:15])
    print(c2_sorted[:15])
    print(c3_sorted[:15])
    plt.show()
    print('done')
