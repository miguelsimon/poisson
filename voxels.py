import unittest
from typing import NamedTuple

import numpy as np
from numpy import array


class Voxels(NamedTuple):
    bounds: np.ndarray
    bins: np.ndarray


def get_bin_edges(bounds, bins):
    """
    Get histogram bin edges for a volume.

    Parameters
    ----------

    bounds : ndarray
        (n, 2) shaped array where each row is [min, max] bounds for that
        dimension
    y : ndarray
        (n,) shaped array of int, each row is the number of voxels to partition
        that dimension into

    Returns
    -------

    List[ndarray]
        A list of n arrays, each containing the bin edges for the corresponding
        dimension

    """
    res = []
    for i in range(len(bins)):
        res.append(np.linspace(bounds[i][0], bounds[i][1], bins[i]))
    return res


def get_voxel_centroids(bin_edges):
    """
    Get centroid arrays for bin edges arrays.

    Parameters
    ----------

    bin_edges : List[ndarray]
        A list of n arrays, each containg bin edges for a dimension

    Returns
    -------

    List[ndarray]
        A list of n arrays, each containing the centroids for the bin edges

    """
    res = []
    for edges in bin_edges:
        centroids = (edges[:-1] + edges[1:]) / 2
        res.append(centroids)
    return res


def to_voxels(bin_edges, weights, coords):
    """
    Voxelize a sparse density.

    Parameters
    ----------

    bin_edges : List[ndarray]
        A list of n arrays, each containing bin edges for a dimension

    weights: ndarray
        (k,) shaped array of weights

    coords: ndarray
        (k, n) shaped array of coordinates; each row is the k'th weight's
        n-dimensional coordinate

    Returns
    -------

    ndarray
        (d1, d2, ..., dn) shaped array of voxel densities
    List[ndarray]
        List of n arrays, each containing the voxel centroids for that dimension

    """
    assert isinstance(coords, np.ndarray)
    assert weights.shape[0] == coords.shape[0]
    assert len(bin_edges) == coords.shape[1]

    centroids = get_voxel_centroids(bin_edges)

    H, _ = np.histogramdd(coords, bins=bin_edges, weights=weights)
    return H, centroids


class Test(unittest.TestCase):
    def test_get_bin_edges(self):
        bounds = array([[0, 1], [0, 10]], dtype=float)
        bins = array([2, 5])
        edges = get_bin_edges(bounds, bins)

        self.assertTrue(np.allclose(edges[0], array([0.0, 1.0])))
        self.assertTrue(np.allclose(edges[1], array([0.0, 2.5, 5.0, 7.5, 10.0])))

    def test_get_voxel_centroids(self):
        edges = array([[0.0, 1.0, 2.0]])
        centroids = get_voxel_centroids(edges)
        self.assertTrue(np.allclose(centroids, [0.5, 1.5]))

    def test_to_voxels_1d(self):
        bin_edges = [array([0.0, 1.0, 2.0])]
        weights = array([1.0, 10.0])
        coords = array([[0.2], [1.5]])

        voxels, centroids = to_voxels(bin_edges, weights, coords)

        self.assertTrue(np.allclose(voxels, [1.0, 10.0]))
        self.assertTrue(np.allclose(centroids, [0.5, 1.5]))
