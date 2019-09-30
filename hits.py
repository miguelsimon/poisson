import unittest
from typing import Tuple

import numpy as np
from numpy import ndarray
from pandas import DataFrame


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
        hits = DataFrame(
            {
                "event_id": [10020, 10020],
                "x": [-89.604973, -77.158882],
                "y": [-140.010086, -151.419952],
                "z": [-36.768826, -35.151833],
                "time": [0.587584, 0.644163],
                "energy": [0.000068, 0.005417],
            }
        )

        print(hits_to_density(hits))
