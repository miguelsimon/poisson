import unittest
from typing import Tuple

import autograd.numpy as np
import scipy.optimize
from autograd import grad
from autograd import hessian_vector_product as hvp

import load_from_npz


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

    def sample(self, mag: float):
        x = mag * np.random.multinomial(1, np.ones(self.x_dim) / self.x_dim)
        Ey = self.obj.Ey_given_x(self.theta, x)
        y = np.random.poisson(Ey)
        return x, y

    def sample_n(self, n: int, mag: float) -> Tuple[np.ndarray, np.ndarray]:
        xs, ys = [], []
        for _ in range(n):
            x, y = self.sample(mag)
            xs.append(x)
            ys.append(y)

        return np.array(xs), np.array(ys)


def solve(theta, y, mag_lb, mag_ub, true_x=None):
    theta_shape = theta.shape
    x_dim = theta_shape[1] - 1

    def objective(x):
        prod = np.dot(theta[:, :-1], x) + theta[:, -1]
        loss = y * np.log(prod) - prod
        return -np.sum(loss)

    x0 = mag_lb * np.ones(x_dim) / np.linalg.norm(np.ones(x_dim))

    jac = grad(objective)

    hvp(objective)

    bounds = scipy.optimize.Bounds(lb=np.zeros(x_dim), ub=np.inf * np.ones(x_dim))

    # constraints = [
    #     scipy.optimize.LinearConstraint(A=np.ones(x_dim), lb=mag_lb, ub=mag_ub)
    # ]

    def lb_constraint(x):
        return np.linalg.norm(x) - mag_lb

    def ub_constraint(x):
        return mag_ub - np.linalg.norm(x)

    constraints = [
        {"type": "eq", "fun": lb_constraint, "jac": grad(lb_constraint)},
        # {"type": "eq", "fun": ub_constraint, "jac": grad(ub_constraint)},
    ]

    def callback(xk, *args):
        if true_x is not None:
            print(np.linalg.norm(xk), np.linalg.norm(true_x))
            print(np.linalg.norm(true_x - xk) / np.linalg.norm(true_x))
            print()

    sol = scipy.optimize.minimize(
        objective,
        x0,
        method="SLSQP",
        jac=jac,
        bounds=bounds,
        constraints=constraints,
        callback=callback,
    )
    return sol["x"], sol


def solve_l1(theta, y, mag_lb, mag_ub, true_x=None, k=0.01):
    theta_shape = theta.shape
    x_dim = theta_shape[1] - 1

    x0 = mag_lb * np.ones(x_dim) / np.linalg.norm(np.ones(x_dim))

    def objective(x):
        prod = np.dot(theta[:, :-1], x) + theta[:, -1]
        loss = y * np.log(prod) - prod
        return -np.sum(loss)

    def transform(x_a, x_b):
        return x_a - x_b

    def untrans(x_trans):
        return x_trans[:x_dim] - x_trans[x_dim:]

    def l1_penalty(x_a, x_b):
        return np.sum(x_a + x_b)

    def regularized_objective(k, x_a, x_b, objective):
        return objective(transform(x_a, x_b)) + k * l1_penalty(x_a, x_b)

    x_a = x0.copy()
    x_b = np.zeros(x_dim)

    x0_trans = np.hstack([x_a, x_b])

    fun = lambda x_trans: regularized_objective(
        k, x_trans[:x_dim], x_trans[x_dim:], objective
    )

    jac = grad(fun)

    #    hessp = hvp(fun)

    bounds = scipy.optimize.Bounds(
        lb=np.zeros(x0_trans.shape), ub=np.inf * np.ones(x0_trans.shape)
    )

    def callback(xk, *args):
        xk = untrans(xk)
        if true_x is not None:
            print(np.linalg.norm(xk), np.linalg.norm(true_x))
            print(np.linalg.norm(true_x - xk) / np.linalg.norm(true_x))
            print()

    sol = scipy.optimize.minimize(
        fun, x0_trans, method="SLSQP", jac=jac, bounds=bounds, callback=callback
    )
    return untrans(sol["x"]), sol


class Test(unittest.TestCase):
    def dont_test(self):
        theta = np.array(
            [
                [1.0, 0.5, 0.1, 0.1, 0.01],
                [0.5, 0.1, 1.0, 0.1, 0.01],
                [0.1, 1.0, 0.5, 0.1, 0.3],  # this one's dcr is high
            ]
        )

        sim = Sim(theta)

        x, y = sim.sample(100)
        x_est, sol = solve_l1(theta, y, 100, 100)
        print(sol)
        print(np.linalg.norm(x - x_est) / np.linalg.norm(x))

    def test2(self):
        theta = load_from_npz.get_theta(0.0001)

        sim = Sim(theta)

        x, y = sim.sample(600000)
        print(np.sum(y))
        x_est, sol = solve_l1(theta, y, 600000, 600000, true_x=x)
        print(sol)
        print(np.linalg.norm(x - x_est) / np.linalg.norm(x))


def get_train_set():
    theta = load_from_npz.get_theta(0.0001)
    sim = Sim(theta)
    xs, ys = sim.sample_n(1000, 60000)
    return theta, xs, ys


if __name__ == "__main__":
    unittest.main()
