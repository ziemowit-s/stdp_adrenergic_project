import argparse
import pickle

import numpy as np

import matplotlib.pyplot as plt
from sklearn.decomposition import NMF
from sklearn.metrics import explained_variance_score
from sklearn.preprocessing import MinMaxScaler

from ltp_models.utils import get_data, filter_by_time, exclude_molecules, agregate_trails, filter_if_all_cols, \
    plot_chart, plot_hitmap

CKp_merge = "CKp CKpCaMCa4"
pPDE4_merge = 'pPDE4 pPDE4cAMP'
Ip35_merge = 'Ip35 Ip35PP1 Ip35PP2BCaMCa4 Ip35PP1PP2BCaMCa4'
pbAR_merge = 'PKAcpbAR PKAcppbAR PKAcpppbAR pbAR ppbAR pppbAR ppppbAR ppppbARGi'
PKAc_merge = 'PKAcISObAR PKAcpISObAR PKAcppISObAR PKAcpppISObAR PKAcbAR PKAcpbAR PKAcppbAR PKAcpppbAR PKAcAMP2 PKAcAMP4 PKAc I1PKAc PKAcNMDAR PKAcPDE4 PKAc_PDE4_cAMP'
S845_merge = 'GluR1_S845 GluR1_S831 GluR1_S845_S831 GluR1_S845_S567 GluR1_S845_CKCaM GluR1_S845_CKpCaM GluR1_S845_CKp GluR1_S845_CKCaM2 GluR1_S845_CKpCaM2 GluR1_S845_CKp2 GluR1_S831_PKAc GluR1_S845_PP1 GluR1_S845_S831_PP1 GluR1_S845_S567_PP1 GluR1_S845_S831_PP1_2 GluR1_S845_S567_PP1_2 GluR1_S831_PP1 GluR1_S845_PP2B GluR1_S845_S831_PP2B GluR1_S845_S567_PP2B'


def get_component_importance(nmf, header, merge_components=None, agregation='max', merge_name_sufix="MERGED"):
    """
    Compute importance of each column for each component in NMF. Then agregates results by max or avg.
    :param nmf:
    :param header:
    :param merge_components:
        lists or str separated by spaces which express molecules to merge.
    :param agregation:
        max or avg. max means: get max value for each protein from any component
    :return:
    """
    probas = nmf.components_.T / np.sum(nmf.components_, axis=1)
    all_compounds = np.zeros(probas.shape[1])
    if merge_components:
        for molecules in merge_components:
            if isinstance(molecules, str):
                molecules = molecules.split(' ')
            try:
                compound = np.array([probas[header.index(m), :] for m in molecules])
            except ValueError:
                continue
            compound = np.sum(compound, axis=0)
            all_compounds += compound
            probas = np.concatenate([probas, compound.reshape(1, compound.shape[0])], axis=0)

            shortest_name = molecules[np.argmin([len(m) for m in molecules])]
            compound_name = "%s_%s" % (shortest_name, merge_name_sufix)
            header.append(compound_name)

    if agregation == 'avg':
        probas = np.average(probas, axis=1)
    elif agregation == 'max':  # get max value for each protein from any component
        probas = np.max(probas, axis=1)

    probas = probas / np.sum(probas, axis=0)
    probas = probas / np.sum(probas)
    print('All Compounds explanation by component:', all_compounds)
    return probas


def nmf(data, n_components, norm=True, plot=False):
    """
    Computes Non-Negative Matrix Factorization.
    :param data:
    :param n_components:
        Number of components.
    :param norm:
        If normalize by MinMaxScaler returned components
    :param plot:
        If plot. Default False.
    :return:
    """
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


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)

    ap.add_argument("--filter", type=float, default=0,
                    help="Filter out molecules from component_importance plot if their value for all paradigms is"
                         " lower than this filter. It is a probability distribution between 0-100. ")
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--trials", nargs='+', help="Trial numbers if required. Default: take all trials", default=None,
                    type=int)
    ap.add_argument("--trial_agregation", help="Many trial agregation type: avg, concat. Default: concat", default='avg')
    ap.add_argument("--component_agregation", help="Agregate protein importance by component: avg, max", default='max')
    args = ap.parse_args()

    with open(args.pickle, "rb") as f:
        dataset = pickle.load(f)

    # Prepare data
    all_probas = []
    all_paradigm_names = []
    for data, header, label, paradigm_name in dataset:
        print(paradigm_name)
        data = agregate_trails(data, agregation=args.trial_agregation)

        # Dim reduction
        nmf_f, nmf_c, explained_variance = nmf(data, args.component_number, plot=False)
        # Get probability importabce by molecule
        probas = get_component_importance(nmf_f, header, agregation=args.component_agregation,
                                          merge_components=[Ip35_merge, CKp_merge, pPDE4_merge, PKAc_merge, pbAR_merge])

        all_probas.append(probas)
        all_paradigm_names.append("%s:%s" % (paradigm_name, label.upper()))
        print('NMF Explained Variance:', explained_variance)

    all_probas = np.array(all_probas) * 100
    if args.filter:
        all_probas, header = filter_if_all_cols(lower_than=args.filter, data=all_probas, header=header)
    plot_chart(data=all_probas, labels=all_paradigm_names, x_names=header)
    plot_hitmap(data=all_probas, y_names=all_paradigm_names, x_names=header)

    plt.show()
