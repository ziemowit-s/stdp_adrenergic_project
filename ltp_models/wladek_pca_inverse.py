import numpy as np

from sklearn.decomposition import NMF, FastICA, PCA


def decomp(mix_met, component=0, method=2, comps=3):
    if method == 0:
        nmf = NMF(n_components=8, init='nndsvd')  # , random_state=0)
    elif method == 1:
        nmf = FastICA(n_components=comps, random_state=0)
        nmf_1 = nmf.fit_transform(mix_met)
        A = nmf.mixing_
        A1 = A[:, 0].reshape(len(A[:, 0]), 1)
        A2 = A[:, 1].reshape(len(A[:, 0]), 1)
        if comps > 2: A3 = A[:, 2].reshape(len(A[:, 0]), 1)
    else:
        nmf = PCA(n_components=comps)
        nmf.fit(mix_met)
        A = nmf.components_
        A1 = A[0, :].reshape(len(A[0, :]), 1)
        A2 = A[1, :].reshape(len(A[1, :]), 1)
        A3 = A[2, :].reshape(len(A[2, :]), 1)
    nmf_1 = nmf.fit_transform(mix_met)
    sygnal = nmf_1[:, component]
    frst_comp = np.dot(nmf_1[:, 0].reshape(len(nmf_1), 1), A1.T)
    scnd_comp = np.dot(nmf_1[:, 1].reshape(len(nmf_1), 1), A2.T)
    if comps > 2: thrd_comp = np.dot(nmf_1[:, 2].reshape(len(nmf_1), 1), A3.T)
    if comps > 2 and len(nmf_1[:, 0]) < len(A1):
        return sygnal, np.array([frst_comp, scnd_comp, thrd_comp]), np.array(
            [nmf_1[:, 0], nmf_1[:, 1], nmf_1[:, 2]]), np.array([A1, A2, A3])
    elif comps > 2 and len(nmf_1[:, 0]) > len(A1):
        return sygnal, np.array([frst_comp, scnd_comp, thrd_comp]), np.array([A1, A2, A3]), np.array(
            [nmf_1[:, 0], nmf_1[:, 1], nmf_1[:, 2]])
    elif comps == 2 and len(nmf_1[:, 0]) > len(A1):
        return sygnal, np.array([frst_comp, scnd_comp]), np.array([A1, A2]), np.array([nmf_1[:, 0], nmf_1[:, 1]])
    else:
        return sygnal, np.array([frst_comp, scnd_comp]), np.array([nmf_1[:, 0], nmf_1[:, 1]]), np.array([A1, A2])
