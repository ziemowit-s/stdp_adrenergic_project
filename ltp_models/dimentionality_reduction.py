import argparse
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler
from ltp_models.utils import get_data

CKp_merge = "CKp CKpCaMCa4"
pPDE4_merge = 'pPDE4 pPDE4cAMP'
Ip35_merge = 'Ip35 Ip35PP1 Ip35PP2BCaMCa4 Ip35PP1PP2BCaMCa4'
pbAR_merge = 'PKAcpbAR PKAcppbAR PKAcpppbAR pbAR ppbAR pppbAR ppppbAR ppppbARGi'
PKAc_merge = 'PKAcISObAR PKAcpISObAR PKAcppISObAR PKAcpppISObAR PKAcbAR PKAcpbAR PKAcppbAR PKAcpppbAR PKAcAMP2 PKAcAMP4 PKAc I1PKAc PKAcNMDAR GluR1_PKAc GluR1_S831_PKAc GluR1_S567_PKAc PKAcPDE4 PKAc_PDE4_cAMP'
S845_merge = 'GluR1_S845 GluR1_S831 GluR1_S845_S831 GluR1_S845_S567 GluR1_S845_CKCaM GluR1_S845_CKpCaM GluR1_S845_CKp GluR1_S845_CKCaM2 GluR1_S845_CKpCaM2 GluR1_S845_CKp2 GluR1_S831_PKAc GluR1_S845_PP1 GluR1_S845_S831_PP1 GluR1_S845_S567_PP1 GluR1_S845_S831_PP1_2 GluR1_S845_S567_PP1_2 GluR1_S831_PP1 GluR1_S845_PP2B GluR1_S845_S831_PP2B GluR1_S845_S567_PP2B'


def plot_component_importance(nmf, header, merge_components=None, avg=False, paradigm_name=None):
    probas = nmf.components_.T / np.sum(nmf.components_, axis=1) * 100
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

    x = np.arange(len(header))
    plt.xticks(x, header, rotation=90)

    if avg:
        probas = np.average(probas, axis=1)
        plt.plot(x, probas, label=paradigm_name)
    else:
        for i in range(0, probas.shape[1]):
            plt.plot(x, probas[:, i], label="comp_%s_%s" % (i, paradigm_name if paradigm_name else ''))
    plt.legend(loc='best')

    print('All Compounds explanation by component:', all_compounds)


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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", nargs='+', required=True)
    ap.add_argument("--morphology", required=True)
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--trials", nargs='+', help="Trial numbers if required. Default: take all trials", default=None, type=int)
    ap.add_argument("--agregation", help="Many trial agregation type: avg, concat. Default: concat", default='concat')
    args = ap.parse_args()

    # Prepare data
    for paradigm in args.prefix:
        data, header, paradigm_name = get_data(prefix=paradigm, trials=args.trials, morpho=args.morphology)
        data = agregate_trails(data, agregation=args.agregation)
        data, header = exclude(data, header, exact=['time', 'Ca', 'Leak'], wildcard=['out', 'buf'])
        data = data[:1000, :]  # compute only for 1000 steps

        # Make FIG 1
        #plt.figure(1)
        nmf, nmf_c, explained_variance = nmf(data, args.component_number, plot=False)
        #plot("CKp_merge", molecules=CKp_merge, data=data, header=header, norm=True)
        #plot("pNMDAR", molecules="pNMDAR", data=data, header=header, norm=True)
        #plot("Ip35_merge", molecules=Ip35_merge, data=data, header=header, norm=True)
        #plot("pPDE4_merge", molecules=pPDE4_merge, data=data, header=header, norm=True)
        #plot("S845_merge", molecules=S845_merge, data=data, header=header, norm=True)
        #lot("PKAc_merge", molecules=PKAc_merge, data=data, header=header, norm=True)
        #plot("pbAR_merge", molecules=pbAR_merge, data=data, header=header, norm=True)
        #plt.legend(loc='best')

        # Make FIG 2
        #plt.figure(2)
        plot_component_importance(nmf, header, avg=True, paradigm_name=paradigm_name,
                                  merge_components=[Ip35_merge, CKp_merge, pPDE4_merge, S845_merge, PKAc_merge, pbAR_merge])
        print('NMF Explained Variance:', explained_variance)

    plt.show()
