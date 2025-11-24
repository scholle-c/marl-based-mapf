import numpy as np

from value_map_learner.training_data_generation import (
    create_distance_table,
    create_training_data_for_map,
    get_all_possible_map_positions,
)


def test_get_all_possible_map_positions_returns_accessible_cells():
    grid = np.array([[True, False], [False, True]])
    positions = get_all_possible_map_positions(grid)
    assert set(positions) == {(0, 0), (1, 1)}


def test_create_distance_table_computes_shortest_paths():
    grid = np.array([[True, True], [True, True]])
    distances = create_distance_table(grid, (0, 0))
    expected = np.array([[0.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(distances, expected)


def test_create_training_data_for_map_uses_all_accessible_positions():
    grid = np.array([[True, False], [True, True]])
    inputs, labels = create_training_data_for_map(grid)

    assert inputs.shape == (3, 3, 2, 2)  # 3 accessible cells, 3 channels
    assert labels.shape == (3, 2, 2)

    # Map channel should equal the original grid (encoded as int8)
    np.testing.assert_array_equal(inputs[0, 0, :, :], grid.astype(np.int8))
    # Obstacles should be marked unreachable in all labels
    np.testing.assert_array_equal(labels[:, 0, 1], grid.size)
    # Goal position always has distance 0
    for goal_idx, goal in enumerate([(0, 0), (1, 0), (1, 1)]):
        assert labels[goal_idx][goal] == 0


def test_create_training_data_for_map_respects_sample_limit():
    grid = np.array([[True, True], [True, True]])
    inputs, labels = create_training_data_for_map(grid, n_samples=2)

    assert inputs.shape[0] == 2
    assert labels.shape[0] == 2
    # Each sample should have exactly one goal and one start
    goal_counts = inputs[:, 1].reshape(2, -1).sum(axis=1)
    start_counts = inputs[:, 2].reshape(2, -1).sum(axis=1)
    np.testing.assert_array_equal(goal_counts, np.ones_like(goal_counts))
    np.testing.assert_array_equal(start_counts, np.ones_like(start_counts))
