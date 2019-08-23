import os
import numpy as np


def get_data(prefix, morpho, trials=None):
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
    header = None
    path_list = prefix.split(os.path.sep)
    folder = os.path.sep.join(path_list[:-1])
    prefix = path_list[-1]
    for file in os.listdir(folder):
        if file.startswith(prefix) and file.endswith('%s.txt' % morpho):
            if trials is not None and len([t for t in trials if 'trial%s' % t in file]) == 0:
                continue
            path = os.path.join(folder, file) if folder is not None else file
            d = np.loadtxt(path, skiprows=1)
            with open(path) as f:
                header = f.readline().split()
            data.append(d)
    return np.array(data), header, prefix
