import unittest
from typing import Tuple

import autograd.numpy as np
import scipy.optimize

import autograd_objective


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


class FitTest(unittest.TestCase):
    def fit(self, theta):
        sim = Sim(theta)
        norms = []
        samples = []
        print()
        for num in [10, 90, 900]:
            samples.extend([sim.sample() for _ in range(num)])
            obj = autograd_objective.Objective(sim.x_dim, sim.y_dim, samples)
            theta_calculated, _sol = fit(obj)
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
