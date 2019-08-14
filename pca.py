import argparse
import os

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def get_data(folder, prefix, morpho):
    result = []
    for file in os.listdir(folder):
        if file.startswith(prefix) and file.endswith('%s.txt' % morpho):
            path = os.path.join(args.folder, file) if args.folder is not None else file
            d = np.loadtxt(path, skiprows=1)
            with open(path) as f:
                header = f.readline().split()
            result.append(d)
    return header, np.array(result)


# idx = header.index("Ca")
# d[:, idx]
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--components", required=True, type=int)
    ap.add_argument("--morf", nargs='+', required=True)
    args = ap.parse_args()

    for m in args.morf:
        header, data = get_data(folder=args.folder, prefix=args.prefix, morpho=m)
        data = np.average(data, axis=0)

        data = StandardScaler().fit_transform(data)

        pca = PCA(n_components=args.components)
        principalComponents = pca.fit_transform(data)
        print(m, sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_[:3])
