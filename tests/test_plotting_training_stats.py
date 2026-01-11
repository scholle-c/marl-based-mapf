import marl_path.model.plotting as plotting
from marl_path.model.stats import TrainingStats
from test_saving_loading_training_stats import DATA_FOLDER
import os


def test_finding_json():
    stats = TrainingStats()
    stats.record_epoch(
        train_loss=0.5,
        soc=10,
        val_loss=0.6,
        soc_model=12,
        soc_no_model=11,
    )

    stats_filename = "temp_plot_training_stats.json"
    stats_filepath = os.path.join(DATA_FOLDER, stats_filename)
    stats._save_as_json(stats_filepath)

    found_files = plotting._find_json_files_in_folder(DATA_FOLDER)
    assert stats_filepath in found_files


def test_loading_training_stats():
    stats = TrainingStats()
    stats.record_epoch(
        train_loss=0.5,
        soc=10,
        val_loss=0.6,
        soc_model=12,
        soc_no_model=11,
    )

    stats_filename = "temp_plot_training_stats.json"
    stats_filepath = os.path.join(DATA_FOLDER, stats_filename)
    stats._save_as_json(stats_filepath)

    training_stats = plotting._load_stats([DATA_FOLDER])
    assert len(training_stats) > 0
