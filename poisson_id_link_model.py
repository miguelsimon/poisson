import unittest
from typing import Tuple

import autograd.numpy as np
import scipy.optimize

import autograd_objective
import evaluate_errors
from evaluate_errors import Estimator


class Problem:
    """
    This is physically more plausible: sensor counts are linear in event counts
    """

    y_shape: Tuple[int]
    x_shape: Tuple[int]
    theta_shape: Tuple[int, int]

    def __init__(self, x_dim, y_dim):
        self.x_shape = (x_dim,)
        self.y_shape = (y_dim,)
        self.theta_shape = (y_dim, x_dim + 1)

    def Ey_given_x(self, theta, x):
        assert theta.shape == self.theta_shape
        assert x.shape == self.x_shape
        assert np.all(theta >= 0)
        return np.dot(theta, np.hstack([x, [1]]))


class Sim:
    def __init__(self, theta):
        self.x_dim = theta.shape[1] - 1
        self.y_dim = theta.shape[0]
        self.obj = Problem(self.x_dim, self.y_dim)
        self.theta = theta

    def sample(self):
        evt_count = np.random.randint(0, 2)
        x = np.random.multinomial(evt_count, np.ones(self.x_dim) / self.x_dim)
        Ey = self.obj.Ey_given_x(self.theta, x)
        y = np.random.poisson(Ey)
        return x, y

    def sample_n(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for _ in range(n):
            x, y = self.sample()
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)


def fit(obj):
    sol = scipy.optimize.minimize(
        obj.objective,
        obj.get_x0(),
        method="trust-constr",
        jac=obj.jac,
        hessp=obj.hessp,
        bounds=obj.bounds,
    )

    return obj.reshape(sol["x"]), sol


class Test(unittest.TestCase):
    def test_sim(self):
        theta = np.array([[1, 0]])
        sim = Sim(theta)
        for i in range(5):
            print(sim.sample())

    def test_sim2(self):
        theta = np.array([[1, 1, 0]])
        sim = Sim(theta)
        for i in range(5):
            print(sim.sample())

    def test_sanity(self):
        obj = Problem(1, 1)
        theta = np.array([[1, 0]])

        Ey = obj.Ey_given_x(theta, np.array([1]))
        print(Ey)


class XObjEstimator(Estimator):
    def __init__(self, name, theta):
        self.name = name
        self.theta = theta

    def get_name(self):
        return self.name

    def predict_x(self, y):
        obj = autograd_objective.XObjective(self.theta, y)

        sol = scipy.optimize.minimize(
            obj.objective, obj.get_x0(), method="SLSQP", jac=obj.jac, bounds=obj.bounds
        )
        return sol["x"]


class MAPEstimator(Estimator):
    def __init__(self, name, theta):
        self.name = name
        self.theta = theta
        self.x_dim = theta.shape[1] - 1

    def get_name(self):
        return self.name

    def loss(self, x, y):
        prod = np.dot(self.theta[:, :-1], x) + self.theta[:, -1]
        loss = y * np.log(prod) - prod
        return -np.sum(loss)

    def predict_x(self, y):
        xs = [np.zeros(self.x_dim)]
        for i in range(self.x_dim):
            x = np.zeros(self.x_dim)
            x[i] = 1.0
            xs.append(x)

        losses = []
        for x in xs:
            losses.append(self.loss(x, y))

        i = np.argmin(losses)
        return xs[i]


class SimTest:
    def __init__(self, theta: np.array):
        self.theta = theta

        sim = Sim(self.theta)

        test_xs, test_ys = sim.sample_n(100)

        estimators = [
            evaluate_errors.NullaryEstimator(sim),
            XObjEstimator("theta", theta),
            MAPEstimator("MAP", theta),
        ]

        self.evaluation = evaluate_errors.Evaluate(estimators, test_xs, test_ys)


class EvalTest(unittest.TestCase):
    def test(self):
        theta = 100 * np.array(
            [
                [1.0, 0.5, 0.1, 0.1, 0.01],
                [0.5, 0.1, 1.0, 0.1, 0.01],
                [0.1, 1.0, 0.5, 0.1, 0.3],  # this one's dcr is high
            ]
        )

        SimTest(theta)


class FitTest(unittest.TestCase):
    def fit(self, theta):
        sim = Sim(theta)

        train_xs, train_ys = sim.sample_n(100)

        obj = autograd_objective.ThetaObjective(train_xs, train_ys)
        theta_calculated, _sol = fit(obj)
        frobenius = np.linalg.norm(theta - theta_calculated, ord="fro")
        print("frobenius {0}".format(frobenius))

    def test_fit1(self):
        self.fit(np.array([[50, 5]]))

    def test_fit2(self):
        self.fit(np.array([[100, 1]]))

    def test_fit3(self):
        theta = 100 * np.array(
            [
                [1.0, 0.5, 0.1, 0.1, 0.01],
                [0.5, 0.1, 1.0, 0.1, 0.01],
                [0.1, 1.0, 0.5, 0.1, 0.3],  # this one's dcr is high
            ]
        )
        self.fit(theta)
