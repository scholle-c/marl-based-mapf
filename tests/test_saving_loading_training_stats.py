from marl_path.model.stats import TrainingStats
import os


DATA_FOLDER = "./tests/data"


def test_saving_and_loading_training_stats():
    stats = TrainingStats()
    stats.record_epoch(
        train_loss=0.5,
        soc=10,
        val_loss=0.6,
        soc_model=12,
        soc_no_model=11,
    )
    stats.record_epoch(
        train_loss=0.4,
        soc=9,
        val_loss=0.5,
        soc_model=11,
        soc_no_model=10,
    )

    temp_filename = "temp_training_stats.json"
    temp_filepath = os.path.join(DATA_FOLDER, temp_filename)
    stats._save_as_json(temp_filepath)

    loaded_stats = TrainingStats.load_from_json(temp_filepath)

    assert loaded_stats.epochs == stats.epochs
    assert loaded_stats.training_loss == stats.training_loss
    assert loaded_stats.validation_loss == stats.validation_loss
    assert loaded_stats.socs == stats.socs
    assert loaded_stats.socs_model == stats.socs_model
    assert loaded_stats.socs_no_model == stats.socs_no_model
