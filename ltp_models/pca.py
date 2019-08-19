import argparse

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# idx = header.index("Ca")
# d[:, idx]
from ltp_models.utils import get_data

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", default=None)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--component_number", required=True, type=int)
    ap.add_argument("--morphology", nargs='+', required=True)
    ap.add_argument("--trial_agregation", help="avg or concat. default=concat", default='avg')

    args = ap.parse_args()

    for m in args.morphology:
        header, data = get_data(folder=args.folder, prefix=args.prefix, morpho=m)
        data = data.T
        
        if args.trial_agregation == "concat":
            data = data.reshape([data.shape[0], data.shape[1]*data.shape[2]])
        elif args.trial_agregation == "avg":
            data = np.average(data, axis=0)
        else:
            raise NameError("Allowed trial agregation are: 'avg', 'concat'.")

        data = StandardScaler().fit_transform(data)

        pca = PCA(n_components=args.component_number)
        principalComponents = pca.fit_transform(data)
        print(m, sum(pca.explained_variance_ratio_), pca.explained_variance_ratio_[:3])
