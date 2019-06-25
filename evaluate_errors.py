from typing import List

import autograd.numpy as np
from matplotlib import pyplot as plt


class Estimator:
    def get_name(self) -> str:
        raise NotImplementedError

    def predict_x(self, y: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class NullaryEstimator(Estimator):
    """
    Predict x by drawing from the marginal x distribution
    """

    def __init__(self, sim):
        self.sim = sim

    def get_name(self):
        return "nullary"

    def predict_x(self, y):
        x, _ = self.sim.sample()
        return x


class Eval:
    def __init__(self, estimator: Estimator, xs: np.ndarray, ys: np.ndarray):
        self.estimator = estimator
        self.xs = xs
        self.ys = ys
        self.x_preds = np.array([estimator.predict_x(y) for y in ys])

        self.errors = np.linalg.norm(self.xs - self.x_preds, axis=1)


class Evaluate:
    def __init__(self, estimators: List[Estimator], xs: np.ndarray, ys: np.ndarray):
        self.evals = [Eval(estimator, xs, ys) for estimator in estimators]

    def boxplot(self):
        data = []
        labels = []
        for ev in self.evals:
            data.append(ev.errors)
            labels.append(ev.estimator.get_name())

        plt.boxplot(data, labels=labels)
        plt.violinplot(data)
