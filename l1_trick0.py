import unittest

import autograd.numpy as np
import numpy
import scipy.optimize
from autograd import grad
from autograd import hessian_vector_product as hvp


def transform(x_a, x_b):
    return x_a - x_b


def l1_penalty(x_a, x_b):
    return np.sum(x_a + x_b)


def regularized_objective(k, x_a, x_b, objective):
    return objective(transform(x_a, x_b)) + k * l1_penalty(x_a, x_b)


def solve(objective, x0, k):
    n = x0.shape[0]

    x_a = x0.copy()
    x_b = np.zeros(n)

    x0_trans = np.hstack([x_a, x_b])

    fun = lambda x_trans: regularized_objective(k, x_trans[:n], x_trans[n:], objective)

    jac = grad(fun)

    hessp = hvp(fun)

    bounds = scipy.optimize.Bounds(
        lb=np.zeros(x0_trans.shape), ub=np.inf * np.ones(x0_trans.shape)
    )

    sol = scipy.optimize.minimize(
        fun, x0_trans, method="trust-constr", jac=jac, hessp=hessp, bounds=bounds
    )

    x = sol["x"]

    return x[:n] - x[n:], sol


class Test(unittest.TestCase):
    def test_minimize_l2(self):
        x0 = np.ones(10)
        objective = lambda x: np.sum(x * x)
        for k in [0.0, 1.0, 10.0, 100.0]:
            res, sol = solve(objective, x0, k)

            numpy.testing.assert_array_almost_equal(res, np.zeros(10), decimal=4)

    def test_minimize(self):
        target = np.array([1.0, 0.0, 0.0])

        def objective(x):
            delta = target - x
            return np.sum(delta * delta)

        x0 = np.ones(3)

        k = 0.0

        res, sol = solve(objective, x0, k)

        numpy.testing.assert_array_almost_equal(res, target, decimal=4)
