import unittest
from typing import NamedTuple

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas
from pandas import DataFrame


class MC(NamedTuple):
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
        plt.plot(stats["energy"], stats["charge"], "bx")
        plt.xlabel("energy")
        plt.ylabel("charge")


def load_mc(filename: str):
    f = h5py.File(filename, "r")
    return MC(
        hits=DataFrame(f["MC"]["hits"][:]), waveforms=DataFrame(f["MC"]["waveforms"][:])
    )


class Test(unittest.TestCase):
    def test(self):
        mc = load_mc("full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.001.pet.h5")
        print(mc.get_stats())

        print(mc.get_xyz_bounds())
