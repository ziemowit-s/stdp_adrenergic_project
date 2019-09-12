import os
import numpy as np

import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


CKp_merge = "CKp CKpCaMCa4"
pPDE4_merge = 'pPDE4 pPDE4cAMP'
Ip35_merge = 'Ip35 Ip35PP1 Ip35PP2BCaMCa4 Ip35PP1PP2BCaMCa4'
pbAR_merge = 'PKAcpbAR PKAcppbAR PKAcpppbAR pbAR ppbAR pppbAR ppppbAR ppppbARGi'
PKAc_merge = 'PKAcISObAR PKAcpISObAR PKAcppISObAR PKAcpppISObAR PKAcbAR PKAcpbAR PKAcppbAR PKAcpppbAR PKAcAMP2 PKAcAMP4 PKAc I1PKAc PKAcNMDAR PKAcPDE4 PKAc_PDE4_cAMP'
S845_merge = 'GluR1_S845 GluR1_S831 GluR1_S845_S831 GluR1_S845_S567 GluR1_S845_CKCaM GluR1_S845_CKpCaM GluR1_S845_CKp GluR1_S845_CKCaM2 GluR1_S845_CKpCaM2 GluR1_S845_CKp2 GluR1_S831_PKAc GluR1_S845_PP1 GluR1_S845_S831_PP1 GluR1_S845_S567_PP1 GluR1_S845_S831_PP1_2 GluR1_S845_S567_PP1_2 GluR1_S831_PP1 GluR1_S845_PP2B GluR1_S845_S831_PP2B GluR1_S845_S567_PP2B'


def get_data(prefix, morpho, trials=None, molecules=None):
    """
    Return numpy.array of data with many trials based on the txt result file named.
    Example name is:
    prefix_trialNUM_morpho.txt
    where:
    * prefix - is file name prefix with potential folder, eg. STDP_paradigms/model_STDP_+10_ms
    * NUM - is iterated number of trial (automated by the method) iterated from 0 to n trials
    * morpho - is morphology part of result, eg. head, neck, spine, dendrite.

    Trial order is not guaranted.
    :param prefix:
         model prefix (eg. model_start_..) with folder
    :param morpho:
        morphology name (eg. head, neck, PSD)
    :param trials:
        trials number. Default is None, meaning all trials will be taken
    :return:
        tuple of header names and numpy array of results (np array of particles, values, trials)
    """
    data = []
    molecule_num = len(molecules) if molecules else None

    path_list = prefix.split(os.path.sep)
    folder = os.path.sep.join(path_list[:-1])
    prefix = path_list[-1]
    for file in os.listdir(folder):
        # filter out files
        if file.startswith(prefix) and file.endswith('%s.txt' % morpho):
            if trials is not None and len([t for t in trials if 'trial%s' % t in file]) == 0:
                continue

            # make path
            path = os.path.join(folder, file) if folder is not None else file

            # get header
            with open(path) as f:
                header = f.readline().split()
                header_len = len(header)
                print('molecules:', len(header))

            # filter out trials with less molecules then specified (if molecules specified)
            if molecule_num is not None and header_len < molecule_num:
                print("%s < %s. Trial skipped." % (header_len, molecule_num))
                continue

            # get data
            d = np.loadtxt(path, skiprows=1)
            print('steps:', d.shape[0])

            # data integrity
            if molecules:
                # if molecules specified - ensure integrity of header
                d = d.T
                integrated_d = []
                integrated_header = []
                for i, m in enumerate(molecules):
                    ii = header.index(m)

                    integrated_header.append(header[ii])
                    integrated_d.append(d[ii])
                d = np.array(integrated_d).T
                header = integrated_header

            data.append(d)

    return np.array(data), header, prefix


def agregate_trails(data, agregation):
    """
    Agregate many trials by average or concatenation.
    :param data:
        numpy.array of trialas obtained by get_data() function
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


def filter_by_time(data, num_steps, step_len, time_start):
    """
    Returns data filtered by time len. Filtering done by number of steps from the starting point in time.
    :param data:
        Can be array of rank 2 or 3.
    :param num_steps:
        Specified in steps.
    :param step_len:
        Specified in miliseconds.
    :param time_start:
        Specified in miliseconds.
    :return:
    """
    step_start = round(time_start/step_len)
    if len(data.shape) == 2:
        data = np.array([data])

    result = []
    for d in data:
        d = d[step_start:step_start+num_steps]
        if len(d) < num_steps:
            print("Steps: %s < %s. Trial skipped." % (len(d), num_steps))
            continue
        result.append(d[step_start:step_start+num_steps])
    return np.array(result)


def exclude_molecules(data, header, exact=None, wildcard=None):
    """
    Return data and header after removal of selected molecules from columns.
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
        to_exclude.extend([h for h in header if w.lower() in h.lower()])

    idxs = []
    for m in to_exclude:
        try:
            idxs.append(header.index(m))
        except ValueError:
            continue

    data = np.delete(data, idxs, axis=-1)
    header = np.delete(header, idxs).tolist()

    return data, header


def get_concentration(data, header, molecules, norm=False, sum_many_cols=False):
    """
    Returns concetration of selected molecule or molecules (assuming data param has molecules in columns).
    If many molecules specified - may return them separated or by sum as a single group.
    :param data:
    :param header:
        header file of molecules data array
    :param molecules:
        list or string separated by space of molecule names to get concentration.
    :param norm:
        if normalize by MinMaxScaler. Default False.
    :param sum_many_cols:
        if sum all resulted colums to single column. Default False.
    :return:
    """
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


def filter_if_all_cols(lower_than: float, data, header):
    """
    Remove cols (from data and header) if all values in a column is lower_than some value.
    :param lower_than:
    :param values:
    :param header:
    :return:
    """
    idx = np.argwhere(np.sum(data > lower_than, axis=0) == 0)

    data = np.delete(data, idx, axis=1)
    header = np.delete(header, idx).tolist()
    return data, header


def plot_concentration(name, molecules, data, header, norm=False, sum_many_cols=True):
    """

    :param name:
    :param molecules:
    :param data:
    :param header:
    :param norm:
    :param sum_many_cols:
    :return:
    """
    c = get_concentration(data, header, molecules=molecules, norm=norm, sum_many_cols=sum_many_cols)
    plt.plot(c, label=name)


def plot_chart(data, x_names, labels):
    """
    Plot chart of data. data.shape[1] and labels size must mach.
    :param data:
    :param x_names:
    :param labels:
    :return:
    """
    x = np.arange(len(x_names))
    plt.xticks(x, x_names, rotation=90)
    for name, d in zip(labels, data):
        plt.plot(x, d, label=name)
    plt.legend(loc='best')


def plot_hitmap(data, x_names, y_names):
    """
    Plot hitmap of values inside 2D array.
    :param data:
    :param x_names:
    :param y_names:
    :return:
    """
    fig, ax = plt.subplots()
    cax = ax.matshow(data, cmap=plt.cm.Blues)
    fig.colorbar(cax)
    ax.set_xticks(np.arange(len(x_names)))
    ax.set_xticklabels(x_names, rotation=90)
    ax.set_yticklabels([''] + y_names)
    ax.set_aspect('auto')

    # show_grid
    plt.gca().set_xticks([x - 0.5 for x in plt.gca().get_xticks()][1:], minor='true')
    plt.gca().set_yticks([y - 0.5 for y in plt.gca().get_yticks()][1:], minor='true')
    plt.grid(linestyle='-', linewidth='0.5', color='black', which='minor')


