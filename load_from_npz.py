import autograd.numpy as np


def load():
    a = np.load("../Downloads/probability_matrix_Petalo.npz")
    p = a["Prob"]
    a = p.reshape((30, 7, 7, 441), order="C")
    alpha = a.reshape(1470, 441, order="F")
    return alpha


def get_theta(dcr: float):
    alpha = load()
    return np.append(alpha.transpose(), dcr * np.ones((441, 1)), axis=1)
