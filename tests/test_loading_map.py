from marl_path.shared.mapf_utils import get_grid
from test_saving_loading_training_stats import DATA_FOLDER
import os
import numpy as np
import numpy.testing as npt


def test_loading_map():
    """
    Test loading the tunnel map. This one has the shape (o encodes passable tile,
    x a wall):
    ------
    |oxxx|
    |oooo|
    |oxxx|
    |oxxx|
    |oxxx|
    |oxxx|
    ------

    As a output, a map/np.array of the shape (6, 4) is expected, containing 1/True values
    for passable tiles and 0/False for wall-tiles.
    """
    map_file: str = "tunnel.map"
    map = get_grid(os.path.join(DATA_FOLDER, map_file))

    expected_map = np.array(
        [
            [1, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=bool,
    )

    assert expected_map.shape == map.shape
    npt.assert_equal(map, expected_map)
