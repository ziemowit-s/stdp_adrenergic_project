import argparse
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

import matplotlib.pyplot as plt

from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from ltp_models.utils import get_data

CKp_merge = "CKp CKpCaMCa4"
pPDE4_merge = 'pPDE4 pPDE4cAMP'
Ip35_merge = 'Ip35 Ip35PP1 Ip35PP2BCaMCa4 Ip35PP1PP2BCaMCa4'
pbAR_merge = 'PKAcpbAR PKAcppbAR PKAcpppbAR pbAR ppbAR pppbAR ppppbAR ppppbARGi'
PKAc_merge = 'PKAcISObAR PKAcpISObAR PKAcppISObAR PKAcpppISObAR PKAcbAR PKAcpbAR PKAcppbAR PKAcpppbAR PKAcAMP2 PKAcAMP4 PKAc I1PKAc PKAcNMDAR PKAcPDE4 PKAc_PDE4_cAMP'
S845_merge = 'GluR1_S845 GluR1_S831 GluR1_S845_S831 GluR1_S845_S567 GluR1_S845_CKCaM GluR1_S845_CKpCaM GluR1_S845_CKp GluR1_S845_CKCaM2 GluR1_S845_CKpCaM2 GluR1_S845_CKp2 GluR1_S831_PKAc GluR1_S845_PP1 GluR1_S845_S831_PP1 GluR1_S845_S567_PP1 GluR1_S845_S831_PP1_2 GluR1_S845_S567_PP1_2 GluR1_S831_PP1 GluR1_S845_PP2B GluR1_S845_S831_PP2B GluR1_S845_S567_PP2B'


def get_component_importance(nmf, header, merge_components=None, agregation='max'):
    """

    :param nmf:
    :param header:
    :param merge_components:
    :param agregation:
        max or avg
    :return:
    """
    probas = nmf.components_.T / np.sum(nmf.components_, axis=1)
    all_compounds = np.zeros(probas.shape[1])
    if merge_components:
        for molecules in merge_components:
            if isinstance(molecules, str):
                molecules = molecules.split(' ')
            compound = np.array([probas[header.index(m), :] for m in molecules])
            compound = np.sum(compound, axis=0)
            all_compounds += compound
            probas = np.concatenate([probas, compound.reshape(1, compound.shape[0])], axis=0)

            compound_name = "%s_merged" % molecules[np.argmin([len(m) for m in molecules])]
            header.append(compound_name)

    if agregation == 'avg':
        probas = np.average(probas, axis=1)
    elif agregation == 'max':
        probas = np.max(probas, axis=1)

    probas = probas/np.sum(probas, axis=0)
    probas = probas / np.sum(probas)
    print('All Compounds explanation by component:', all_compounds)
    return probas


def filter_time(data, num_steps, step_len, time_start):
    step_start = round(time_start/step_len)
    if num_steps:
        return data[step_start:step_start+num_steps]
    else:
        return data[step_start:]


def plot(name, molecules, data, header, norm=False, sum_many_cols=True):
    c = get_concentration(data, header, molecules=molecules, norm=norm, sum_many_cols=sum_many_cols)
    plt.plot(c, label=name)


def nmf(data, n_components, norm=True, plot=True):
    if norm:
        data = MinMaxScaler().fit_transform(data)

    nmf = NMF(n_components=n_components)
    c = nmf.fit_transform(data)

    predictions = nmf.inverse_transform(c)
    explained_variance = explained_variance_score(data, predictions)

    if norm:
        c = MinMaxScaler().fit_transform(c)

    if plot:
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

    idxs = []
    for m in to_exclude:
        try:
            idxs.append(header.index(m))
        except ValueError:
            continue

    data = np.delete(data, idxs, axis=1)
    header = np.delete(header, idxs).tolist()

    return data, header


def get_concentration(data, header, molecules, norm=False, sum_many_cols=False):
    if isinstance(molecules, str):
        molecules = molecules.split(' ')

    v = np.array([data[:, header.index(m)] for m in molecules]).T

    if sum_many_cols:
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


def plot_probas_chart(values, names, header):
    x = np.arange(len(header))
    plt.xticks(x, header, rotation=90)
    for name, probas in zip(names, values):
        plt.plot(x, probas, label=name)
    plt.legend(loc='best')


def plot_probas_hitmap(values, names, header):
    fig, ax = plt.subplots()
    cax = ax.matshow(values, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(header)))
    ax.set_xticklabels(header, rotation=90)
    ax.set_yticklabels([''] + names)
    ax.set_aspect('auto')

    # show_grid
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(linestyle='-', linewidth='0.5', color='black', which='minor')


def filter_if_all_cols(lower_than: float, values, header):
    idx = np.argwhere(np.sum(values > lower_than, axis=0) == 0)

    values = np.delete(values, idx, axis=1)
    header = np.delete(header, idx).tolist()
    return values, header


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs='+', required=True)

    ap.add_argument("--time_start", nargs='+', required=True, type=int, help="in ms")
    ap.add_argument("--num_steps", type=int)
    ap.add_argument("--step_len", type=int, help="in ms")

    ap.add_argument("--molecule_num", type=int)
    ap.add_argument("--filter", type=float, default=None)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--trials", nargs='+', help="Trial numbers if required. Default: take all trials", default=None, type=int)
    ap.add_argument("--agregation", help="Many trial agregation type: avg, concat. Default: concat", default='concat')
    args = ap.parse_args()

    # Prepare data
    all_probas = []
    all_paradigm_names = []
    for paradigm, time_start in list(zip(args.prefix, args.time_start))[:2]:
        print(paradigm)
        data, header, paradigm_name = get_data(prefix=paradigm, trials=args.trials,
                                               morpho=args.morphology, molecule_num=args.molecule_num)
        data = agregate_trails(data, agregation=args.agregation)
        data, header = exclude(data, header,
                               exact=['time', 'Ca', 'Leak'] + S845_merge.split(' '),
                               wildcard=['out', 'buf'])

        data = filter_time(data, num_steps=args.num_steps, step_len=args.step_len, time_start=time_start)

        nmf_f, nmf_c, explained_variance = nmf(data, args.component_number, plot=False)
        #plt.figure(1)
        #plot("CKp_merge", molecules=CKp_merge, data=data, header=header, norm=True)
        #plt.legend(loc='best')

        probas = get_component_importance(nmf_f, header,
                                          merge_components=[Ip35_merge, CKp_merge, pPDE4_merge, PKAc_merge, pbAR_merge])
        all_probas.append(probas)

        all_paradigm_names.append(paradigm_name)
        print('NMF Explained Variance:', explained_variance)

    all_probas = np.array(all_probas) * 100
    if args.filter:
        all_probas, header = filter_if_all_cols(lower_than=args.filter, values=all_probas, header=header)
    plot_probas_chart(values=all_probas, names=all_paradigm_names, header=header)
    plot_probas_hitmap(values=all_probas, names=all_paradigm_names, header=header)

    plt.show()
