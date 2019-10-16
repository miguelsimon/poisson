import unittest
from typing import NamedTuple, Tuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas
from numpy import ndarray
from pandas import DataFrame

Bounds = Tuple[Tuple[float, float], Tuple[float, float]]


def get_aspect_bounds(x: ndarray, y: ndarray) -> Bounds:
    """
    Return an aspect radio preserving bounding box for

    Parameters
    ----------

    x : ndarray
        (1,) shaped array of x coordinates
    y : ndarray
        (1,) shaped array of y coordinates

    Returns
    -------

    Bounds
        ((x_min, x_max), (y_min, y_max))
    """

    x_ext = x.max() - x.min()
    y_ext = y.max() - y.min()

    mid_x = (x.min() + x.max()) / 2
    mid_y = (y.min() + y.max()) / 2

    ext = max(x_ext, y_ext)
    half = ext / 2

    return (mid_x - half, mid_x + half), (mid_y - half, mid_y + half)


class HitStats:
    def __init__(self, all_hits: DataFrame):
        self.hits = all_hits

        groups = self.hits.groupby("event_id")
        self.dfs = [df for _, df in groups]

    def _scatter(self, x, y, x_bounds, y_bounds):
        plt.figure()
        plt.scatter(x, y)
        plt.xlim(*x_bounds)
        plt.ylim(*y_bounds)

    def get_aspect_bounds(self, x_coord: str, y_coord: str) -> Bounds:
        x = np.array(self.hits[x_coord])
        y = np.array(self.hits[y_coord])

        return get_aspect_bounds(x, y)

    def scatter(self, hits: DataFrame, x_coord: str, y_coord: str, zoom: bool):
        x = np.array(hits[x_coord])
        y = np.array(hits[y_coord])

        if zoom:
            x_bounds, y_bounds = get_aspect_bounds(x, y)
        else:
            x_bounds, y_bounds = self.get_aspect_bounds(x_coord, y_coord)

        self._scatter(x, y, x_bounds, y_bounds)
        plt.xlabel(x_coord)
        plt.ylabel(y_coord)

    def scatter3d(self, hits: DataFrame):
        x = np.array(hits["x"])
        y = np.array(hits["y"])
        z = np.array(hits["z"])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(x, y, z)


class Stats(NamedTuple):
    hits: DataFrame
    waveforms: DataFrame

    def get_stats(self):
        ghits = self.hits.groupby("event_id")
        gwavs = self.waveforms.groupby("event_id")

        energy = ghits["energy"].sum()
        deposition_count = ghits.size().rename("deposition_count")
        charge = gwavs["charge"].sum()
        sensor_count = gwavs.size().rename("sensor_count")

        return pandas.concat([energy, deposition_count, charge, sensor_count], axis=1)

    def get_xyz_bounds(self):
        res = []
        for coord in "x y z".split():
            res.append([self.hits[coord].min(), self.hits[coord].max()])
        return np.array(res)

    def draw_hists(self):
        stats = self.get_stats()
        plt.figure()
        plt.subplot(2, 2, 1)
        plt.hist(stats["energy"])
        plt.title("energy")

        plt.subplot(2, 2, 2)
        plt.hist(stats["deposition_count"])
        plt.title("deposition_count")

        plt.subplot(2, 2, 3)
        plt.hist(stats["charge"])
        plt.title("charge")

        plt.subplot(2, 2, 4)
        plt.hist(stats["sensor_count"])
        plt.title("sensor_count")

        plt.tight_layout()

    def draw_energy_vs_charge(self):
        stats = self.get_stats()
        plt.figure()
        plt.plot(stats["energy"], stats["charge"], "bx")
        plt.xlabel("energy")
        plt.ylabel("charge")


def load_mc(filename: str):
    with h5py.File(filename, "r") as f:
        return Stats(
            hits=DataFrame(f["MC"]["hits"][:]),
            waveforms=DataFrame(f["MC"]["waveforms"][:]),
        )


class Test(unittest.TestCase):
    def test(self):
        mc = load_mc("full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.001.pet.h5")
        print(mc.get_stats())

        print(mc.get_xyz_bounds())
