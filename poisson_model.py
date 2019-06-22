import unittest
from typing import Tuple

import autograd.numpy as np
import scipy.optimize
from autograd import grad
from autograd import hessian_vector_product as hvp


class Problem:
    """
    Following the treatment here: https://en.wikipedia.org/wiki/Poisson_regression#Regression_models
    """

    y_shape: Tuple[int]
    x_shape: Tuple[int]
    theta_shape: Tuple[int, int]

    def __init__(self, x_dim, y_dim):
        self.x_shape = (x_dim,)
        self.y_shape = (y_dim,)
        self.theta_shape = (y_dim, x_dim + 1)

    def log_Ey_given_x(self, theta, x):
        assert theta.shape == self.theta_shape
        assert x.shape == self.x_shape
        return np.dot(theta, np.hstack([x, [1]]))

    def Ey_given_x(self, theta, x):
        return np.exp(self.log_Ey_given_x(theta, x))


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


def fit(x_dim, y_dim, samples):
    theta_shape = (y_dim, x_dim + 1)
    theta0 = np.zeros(theta_shape).flatten()

    lxs, lys = [], []
    for x, y in samples:
        lxs.append(x)
        lys.append(y)

    xs = np.array(lxs).transpose()
    ys = np.array(lys).transpose()

    def objective(theta_flat):
        theta = np.reshape(theta_flat, theta_shape)

        # just throw np.newaxis in there until it works
        prods = np.dot(theta[:, :-1], xs) + theta[:, -1][:, np.newaxis]
        losses = ys * prods - np.exp(prods)
        return -np.sum(losses)

    jac = grad(objective)

    hessp = hvp(objective)

    bounds = scipy.optimize.Bounds(
        lb=np.zeros(theta0.shape), ub=np.inf * np.ones(theta0.shape)
    )

    sol = scipy.optimize.minimize(
        objective, theta0, method="trust-constr", jac=jac, hessp=hessp, bounds=bounds
    )

    return sol["x"].reshape(theta_shape), sol


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


class FitTest(unittest.TestCase):
    def fit(self, theta):
        sim = Sim(theta)
        norms = []
        samples = []
        print()
        for num in [10, 90, 900]:
            samples.extend([sim.sample() for _ in range(num)])
            theta_calculated, _sol = fit(sim.x_dim, sim.y_dim, samples)
            frobenius = np.linalg.norm(theta - theta_calculated, ord="fro")
            norms.append(frobenius)
            print("samples: {0} frobenius: {1}".format(len(samples), frobenius))

        self.assertTrue(norms[-1] < norms[0])

    def test_fit1(self):
        self.fit(np.array([[1, 5]]))

    def test_fit2(self):
        self.fit(np.array([[1, 1]]))

    def test_fit3(self):
        theta = np.array(
            [
                [1.0, 0.5, 0.1, 0.1, 0.01],
                [0.5, 0.1, 1.0, 0.1, 0.01],
                [0.1, 1.0, 0.5, 0.1, 0.3],  # this one's dcr is high
            ]
        )
        self.fit(theta)
