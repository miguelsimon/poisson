import pprint

import autograd.numpy as np
from autograd import grad, hessian
from cvxopt import log, matrix, solvers, spdiag

import solve_theta

def loss(theta, y, x):
    xs = np.array([x]).transpose()
    prods = np.dot(theta[:, :-1], xs) + theta[:, -1][:, np.newaxis]
    losses = ys.transpose() * np.log(prods) - prods
    return -np.sum(losses)

def dca_iteration(theta, y, xk, alpha, true_x):
    xk_norm = np.linalg.norm(xk)

    if xk_norm > 0:
        linearized_l2 = xk / xk_norm
    else:
        linearized_l2 = np.zeros(xk.shape)

    return poisson_x(theta, y, xk, alpha, linearized_l2, true_x)

def poisson_x(theta, y, xk, alpha, lin_l2, true_x = None):
    ys = np.array([y])

    theta_shape = theta.shape
    x_dim = theta_shape[1] - 1
    y_dim = y.shape[0]

    n = x_dim

    G = spdiag(matrix(-np.ones(n)))
    h = matrix(0.0, (n, 1))

    x0_np = np.array([xk]).transpose().astype('double')
    x0 = matrix(x0_np)

    def objective(x):
        xs = np.array([x]).transpose()
        prods = np.dot(theta[:, :-1], xs) + theta[:, -1][:, np.newaxis]
        losses = ys.transpose() * np.log(prods) - prods
        return -np.sum(losses) + alpha * (np.sum(x) - np.dot(x, lin_l2))

    jac = grad(objective)
    hess = hessian(objective)

    def x_from_matrix(xm):
        return np.array(xm).transpose()[0]

    def obj(x):
        x = x_from_matrix(x)
        res = objective(x)

        return res

    def obj_grad(x):
        x = x_from_matrix(x)
        val = jac(x)
        res = matrix(np.array([val]))
        return res

    def obj_hessian(x):
        x = x_from_matrix(x)
        res = matrix(hess(x))
        return res

    def F(x=None, z=None):
        if x is None:
            return 0, x0
        if z is None:
            return obj(x), obj_grad(x)
        else:
            return obj(x), obj_grad(x), z[0] * obj_hessian(x)

    res = solvers.cp(F, G=G, h=h)["x"]
    return np.array(res).transpose()[0]

def rel_error(x, true_x):
    return np.linalg.norm(true_x - x) / np.linalg.norm(true_x)


def elt_error(x, true_x):
    return np.max(np.abs(x - true_x))

def solve(theta, y, alpha, true_x):
    xk = np.zeros(true_x.shape)
    i = 0
    while 1:

        xk_next = dca_iteration(theta, y, xk.copy(), alpha, true_x)

        data = dict(
                rel_error = rel_error(xk_next, true_x),
                elt_error = elt_error(xk_next, true_x),
                l2_x = np.linalg.norm(xk_next),
                l2_x_true = np.linalg.norm(true_x),
                loss_x = loss(theta, y, xk_next),
                loss_x_true = loss(theta, y, true_x),
                l1_l2_dca_x = alpha * (np.sum(xk_next) - np.linalg.norm(xk_next)),
                l1_l2_dca_x_true = alpha * (np.sum(true_x) - np.linalg.norm(true_x)),
        )
        print()
        print()
        print('------------------------------')
        print('i {0} rel_error xk xk_next {1}'.format(i, rel_error(xk, xk_next)))
        pprint.pprint(data)
        if rel_error(xk, xk_next) < 0.1:
            break
        xk = xk_next

    print('finished, rel_error {0}'.format(rel_error(xk_next, true_x)))
    print('xk_next loss {0}'.format(loss(theta, y, xk_next)))
    print('true_x loss {0}'.format(loss(theta, y, true_x)))

if __name__ == "__main__":
    theta, xs, ys = solve_theta.get_train_set()
    alpha = 0.001

    solve(theta, ys[0], alpha, xs[0])
