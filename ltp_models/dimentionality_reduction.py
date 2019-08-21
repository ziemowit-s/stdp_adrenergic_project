import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

# idx = header.index("Ca")
# d[:, idx]
from ltp_models.utils import get_data


def exclude(data, header, molecules):
    """
    :param data:
    :param header:
    :param molecules:
    :return:
        tuple(data, header) after removal
    """
    idxs = [header.index(m) for m in molecules]
    return np.delete(data, idxs, axis=1), np.delete(header, idxs).tolist()


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

    to_exclude = [h for h in header if "out" in h.lower()]
    to_exclude.append('time')

    data, header = exclude(data, header, molecules=to_exclude)
    print(header)

    f = PCA(n_components=args.component_number)
    c = f.fit_transform(MinMaxScaler().fit_transform(data))

    # plot normalized PCA components
    c = MinMaxScaler().fit_transform(c)
    plt.plot(c[:, 0], label='Component_1')
    plt.plot(c[:, 1], label='Component_2')

    # plot normalized concentrations
    plt.plot(get_concentration(data, header, molecules="CKp CKpCaMCa4", norm=True, sum_cols=True), label='CKp')
    plt.plot(get_concentration(data, header, molecules="pbAR ppbAR pppbAR ppppbAR", norm=True, sum_cols=True), label='pbAR')
    plt.plot(get_concentration(data, header, molecules="PKA", norm=True), label='PKA')

    plt.legend(loc='best')
    plt.show()
    print('done')
