import os
import numpy as np


def get_data(folder, prefix, morpho, no_time=False):
    """

    :param folder:
        folder where all txt files for the model are
    :param prefix:
         model prefix (eg. model_start_..)
    :param morpho:
        morphology name (eg. head, neck, PSD)
    :param no_time:
        remove time column. Default False
    :return:
        tuple of header names and numpy array of results (np array of particles, values, trials)
    """
    data = []
    header = None
    for file in os.listdir(folder):
        if file.startswith(prefix) and file.endswith('%s.txt' % morpho):
            path = os.path.join(folder, file) if folder is not None else file
            d = np.loadtxt(path, skiprows=1)
            with open(path) as f:
                header = f.readline().split()
            data.append(d)
    data = np.array(data).T

    if no_time:
        header = header[1:]
        data = data[1:]
    return header, data
