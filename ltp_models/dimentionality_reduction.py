import argparse
from collections import Iterable

import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from ltp_models.utils import get_data


def plot(name, molecules, data, header):
    c = get_concentration(data, header, molecules=molecules, norm=True, sum_cols=True)
    plt.plot(c, label=name)


def reduce_and_plot_nmf(data, n_components, normalize=True):
    if normalize:
        data = MinMaxScaler().fit_transform(data)

    nmf = NMF(n_components=n_components)
    c = nmf.fit_transform(data)

    predictions = nmf.inverse_transform(c)
    explained_variance = explained_variance_score(data, predictions)

    for i in range(0, c.shape[1]):
        plt.plot(c[:, i], label='%s_%s' % ("NMF", i))

    return nmf, c, explained_variance


def exclude(data, header, exact=None, wildcard=None):
    """

    :param data:
    :param header:
    :param exact:
        to exclude by exact name
    :param wildcard:
        to exclude by name which contains
    :return:
        tuple(data, header) after removal
    """
    if exact is None:
        exact = []
    if wildcard is None:
        wildcard = []

    if isinstance(exact, str):
        exact = exact.split(' ')
    if isinstance(wildcard, str):
        wildcard = wildcard.split(' ')

    to_exclude = exact
    for w in wildcard:
        to_exclude.extend([h for h in header if w in h.lower()])

    idxs = [header.index(m) for m in to_exclude]
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


def agregate_trails(data, agregation):
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

    # Prepare data
    data, header = get_data(folder=args.folder, prefix=args.prefix, trials=args.trials, morpho=args.morphology)
    data = agregate_trails(data, agregation=args.agregation)
    data, header = exclude(data, header, exact=['time', 'Ca', 'Leak'], wildcard=['out', 'buf'])
    data = data[:1000, :]  # compute only for 1000 steps

    # Make FIG 1
    plt.figure(1)
    nmf, nmf_c, explained_variance = reduce_and_plot_nmf(data, args.component_number)
    plot("CK", molecules="CK", data=data, header=header)
    plot("Gi", molecules="Gi", data=data, header=header)
    plot("PKA", molecules="PKA", data=data, header=header)
    plot("CKp", molecules="CKp CKpCaMCa4", data=data, header=header)
    plot("pbAR", molecules="pbAR ppbAR pppbAR ppppbAR", data=data, header=header)
    #plot("Gibg", molecules="Gibg", data=data, header=header)
    #plot("CaMCa", molecules="CaMCa2 CaMCa4", data=data, header=header)
    plt.legend(loc='best')

    # Make FIG 2
    plt.figure(2)
    molecules_by_components = MinMaxScaler().fit_transform(nmf.components_.T)
    x = np.arange(len(header))
    plt.xticks(x, header, rotation=90)
    plt.plot(x, molecules_by_components[:, 0])
    plt.plot(x, molecules_by_components[:, 1])

    # Print 15 the most important molecules for 2 components
    c1_sorted = sorted(zip(header, molecules_by_components[:, 0]), key=lambda x: -x[1])
    c2_sorted = sorted(zip(header, molecules_by_components[:, 1]), key=lambda x: -x[1])
    print(c1_sorted[:15])
    print(c2_sorted[:15])
    print('NMF Explained Variance:', explained_variance)

    plt.show()
