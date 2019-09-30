import unittest
from typing import Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame

import emd


def hits_to_density(hits: DataFrame) -> Tuple[ndarray, ndarray]:
    """
    Extracts a densty as weights and coordinates from a hits DataFrame

    Parameters
    ----------

    hits : DataFrame

    Returns
    -------

    ndarray
        (n,) shaped array of weights
    ndarray
        (n, 3) shaped array of x, y, z coordinates
    """
    weights = np.array(hits["energy"])
    coords = np.array(hits[["x", "y", "z"]])
    return weights, coords


class Test(unittest.TestCase):
    def test(self):
        hits1 = DataFrame(
            {
                "event_id": [10020, 10020],
                "x": [-89.604973, -77.158882],
                "y": [-140.010086, -151.419952],
                "z": [-36.768826, -35.151833],
                "time": [0.587584, 0.644163],
                "energy": [0.000068, 0.005417],
            }
        )

        hits2 = DataFrame(
            {
                "event_id": [10020],
                "x": [-89.604973],
                "y": [-140.010086],
                "z": [-36.768826],
                "time": [0.587584],
                "energy": [0.000068 + 0.005417],
            }
        )

        weights1, points1 = hits_to_density(hits1)

        dist, _ = emd.sparse_emd(weights1, points1, weights1, points1)

        self.assertTrue(np.allclose(dist, 0))

        weights2, points2 = hits_to_density(hits2)

        dist, _ = emd.sparse_emd(weights1, points1, weights2, points2)

        self.assertFalse(np.allclose(dist, 0))
