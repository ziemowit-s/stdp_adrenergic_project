import os
import numpy as np


def get_data(folder, prefix, morpho):
    """

    :param folder:
        folder where all txt files for the model are
    :param prefix:
         model prefix (eg. model_start_..)
    :param morpho:
        morphology name (eg. head, neck, PSD)
    :return:
        tuple of header names and numpy array of results
    """
    result = []
    header = None
    for file in os.listdir(folder):
        if file.startswith(prefix) and file.endswith('%s.txt' % morpho):
            path = os.path.join(folder, file) if folder is not None else file
            d = np.loadtxt(path, skiprows=1)
            with open(path) as f:
                header = f.readline().split()
            result.append(d)
    return header, np.array(result)
