import os
import numpy as np


def get_data(prefix, morpho, trials=None, molecules=None):
    """
    Trial order is not guaranted
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
                print(len(header), 'molecules in trial')

            # filter out trials with less molecules then specified (if molecules specified)
            if molecule_num is not None and header_len < molecule_num:
                print("%s < %s. Trial skipped." % (header_len, molecule_num))
                continue

            # get data
            d = np.loadtxt(path, skiprows=1)

            # data integrity
            if molecules:
                # if molecules specified - ensure integrity of header
                d = d.T
                integrated_d = []
                for i, m in enumerate(molecules):
                    ii = header.index(m)
                    integrated_d.append(d[ii])
                d = np.array(integrated_d).T

            data.append(d)

    return np.array(data), prefix
