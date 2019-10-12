import unittest
from typing import List, Sequence

import h5py
import numpy as np
from numpy import ndarray
from pandas import DataFrame


def get_xyz_bounds(hits):
    res = []
    for coord in "x y z".split():
        res.append([hits[coord].min(), hits[coord].max()])

        # extend max bound so it includes the bound in the open interval
        res[-1][-1] = np.nextafter(res[-1][-1], np.inf)

    return np.array(res)


def get_bin_edges(bounds: ndarray, divisions: Sequence[int]) -> List[ndarray]:
    """
    Get voxel bin edges for a volume.

    Parameters
    ----------

    bounds : ndarray
        (n, 2) shaped array where each row is [min, max] bounds for that
        dimension
    bins : Sequence[int]
        each row is the number of voxels to partition that dimension into

    Returns
    -------

    List[ndarray]
        A list of n arrays, each containing the bin edges for the corresponding
        dimension

    """
    assert bounds.shape[0] == len(divisions)
    res = []
    for i in range(len(divisions)):
        res.append(np.linspace(bounds[i][0], bounds[i][1], divisions[i]))

    return res


def digitize(data, bins: ndarray) -> ndarray:
    idxs = np.digitize(data, bins)

    # there should be nothing outside min bounds
    assert idxs.min() == 1

    # max bound needs to be respected
    assert bins[-1] - data.max() >= 0

    return idxs - 1


class MC:
    """
    Process the MC data such that:
        * events are given explicit indices
        * sensors are given explicit indices
        * x, y, z dimensions are partitioned into voxels and given explicit indices

    This makes it easy to do stuff like list event hits-waveform pairs, build dense arrays etc

    Parameters
    ----------

    f : h5py.File
        file with MC data

    bins : Tuple[ndarray, ndarray, ndarray]
        bins along the x, y, z dimensions

    Attributes
    ----------

    event_idxs : DataFrame
        table of 'event_id', 'event_idx'

    sensor_idxs : DataFrame
        table of 'sensor_id', 'sensor_idx'

    ext_hits : DataFrame
        add 'event_idx', 'ix', 'iy', 'iz' to hits table

    ext_wavs : DataFrame
        add 'event_idx', 'sensor_idx' to waveforms table

    """

    def __init__(self, f, bins):
        waveforms = DataFrame(f["MC"]["waveforms"][:])
        sensor_positions = DataFrame(f["MC"]["sensor_positions"][:])

        sensor_idxs = sensor_positions[["sensor_id"]].copy()
        sensor_idxs["sensor_idx"] = range(len(sensor_idxs))

        hits = DataFrame(f["MC"]["hits"][:])
        all_event_ids = np.hstack(
            [np.array(hits["event_id"]), np.array(waveforms["event_id"])]
        )
        event_ids = np.unique(all_event_ids)

        event_idxs = DataFrame(
            {"event_id": event_ids, "event_idx": range(len(event_ids))}
        )

        ext_hits = hits.join(event_idxs.set_index("event_id"), on="event_id")

        ext_hits["ix"] = digitize(ext_hits["x"], bins[0])
        ext_hits["iy"] = digitize(ext_hits["y"], bins[1])
        ext_hits["iz"] = digitize(ext_hits["z"], bins[2])

        # add sensor idx to waveforms
        ext_waveforms = waveforms.join(
            sensor_idxs.set_index("sensor_id"), on="sensor_id"
        )
        # add event idx to waveforms
        ext_waveforms = ext_waveforms.join(
            event_idxs.set_index("event_id"), on="event_id"
        )

        self.bins = bins

        self.event_idxs = event_idxs
        self.sensor_idxs = sensor_idxs
        self.ext_hits = ext_hits
        self.ext_wavs = ext_waveforms


def file_to_model(filename: str, divisions: Sequence[int]) -> MC:
    with h5py.File(filename, "r") as f:
        hits = DataFrame(f["MC"]["hits"][:])
        bounds = get_xyz_bounds(hits)
        bins = get_bin_edges(bounds, divisions)
        return MC(f, bins)


class Test(unittest.TestCase):
    def test_model(self):
        filename = "full_ring_iradius165mm_depth3cm_pitch7mm_new_h5.001.pet.h5"
        m = file_to_model(filename, (5, 5, 5))
        print(m)
