import autograd.numpy as np
import scipy.optimize
from autograd import grad
from autograd import hessian_vector_product as hvp


class ThetaObjective:
    def __init__(self, x_dim, y_dim, samples):
        theta_shape = (y_dim, x_dim + 1)
        theta0 = np.zeros(theta_shape)
        theta0[:, -1] = 1  # bias
        theta0 = theta0.flatten()

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
            losses = ys * np.log(prods) - prods
            return -np.sum(losses)

        def get_x0():
            return theta0.copy()

        def reshape(theta):
            return theta.reshape(theta_shape)

        self.get_x0 = get_x0

        self.reshape = reshape

        self.objective = objective

        self.jac = grad(objective)

        self.hessp = hvp(objective)

        self.bounds = scipy.optimize.Bounds(
            lb=np.zeros(theta0.shape), ub=np.inf * np.ones(theta0.shape)
        )


class XObjective:
    def __init__(self, theta, y):
        theta_shape = theta.shape
        x_dim = theta_shape[1] - 1

        def objective(x):
            prod = np.dot(theta[:, :-1], x) + theta[:, -1]
            loss = y * np.log(prod) - prod
            return -np.sum(loss)

        def get_x0():
            return np.zeros(x_dim)

        self.get_x0 = get_x0

        self.objective = objective

        self.jac = grad(objective)

        self.hessp = hvp(objective)

        self.bounds = scipy.optimize.Bounds(
            lb=np.zeros(x_dim), ub=np.inf * np.ones(x_dim)
        )
