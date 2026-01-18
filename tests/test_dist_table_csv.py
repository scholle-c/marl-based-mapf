from marl_path.visualization.plotting import load_dist_tables


def test_loading_training_data_history():
    folder_path = "./output/default_output"

    dt_model, dt_lacam = load_dist_tables([folder_path])

    assert len(dt_model) != 0
    assert len(dt_lacam) != 0