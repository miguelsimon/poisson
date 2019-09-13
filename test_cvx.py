import autograd.numpy as np
from autograd import grad, hessian
from cvxopt import log, matrix, solvers, spdiag

import solve_theta


def poisson_x(theta, y, true_x=None):
    ys = np.array([y])

    theta_shape = theta.shape
    x_dim = theta_shape[1] - 1
    y_dim = y.shape[0]

    n = x_dim

    G = spdiag(matrix(-np.ones(n)))
    h = matrix(0.0, (n, 1))

    # A = matrix(np.ones((1, n)))
    # b = matrix(60000 * np.ones((1,1)))
    # x0_np = np.array([true_x]).transpose().astype('double')
    # x0 = matrix(x0_np)

    x0 = matrix(60000.0 / n, (n, 1))

    def objective(x):
        xs = np.array([x]).transpose()
        prods = np.dot(theta[:, :-1], xs) + theta[:, -1][:, np.newaxis]
        losses = ys.transpose() * np.log(prods) - prods
        return -np.sum(losses)

    jac = grad(objective)
    hess = hessian(objective)

    def x_from_matrix(xm):
        return np.array(xm).transpose()[0]

    def obj(x):
        x = x_from_matrix(x)
        res = objective(x)
        print(np.sum(x))
        print(
            "rel: {0} elt: {1} norms: {2} {3} loss: {4} {5}".format(
                rel_error(x, true_x),
                elt_error(x, true_x),
                np.linalg.norm(x),
                np.linalg.norm(true_x),
                res,
                objective(true_x),
            )
        )
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


if __name__ == "__main__":
    theta, xs, ys = solve_theta.get_train_set()
    x_calc = poisson_x(theta, ys[0], xs[0])
