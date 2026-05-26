import os
import tempfile

from marl_path.model.stats import TrainingStats, MAPFStats

DATA_FOLDER = "./tests/data"


def test_saving_and_loading_training_stats():
    stats = TrainingStats(
        training_mode="test",
        used_device="cpu",
        used_seed=42,
        mapf=MAPFStats(num_agents=2, map_size=(4, 4)),
    )
    stats.record_epoch(loss=0.5, soc=10)
    stats.record_epoch(loss=0.4, soc=9)
    stats.record_epoch(loss=None, soc=None)

    with tempfile.TemporaryDirectory() as tmp:
        stats.save(tmp)

        loaded = TrainingStats.load(tmp)

        assert loaded._epoch_count == 3
        assert loaded._losses == [0.5, 0.4, None]
        assert loaded.mapf is not None
        assert loaded.mapf.socs == [10, 9, None]
        assert loaded.mapf.num_agents == 2
        assert loaded.mapf.map_size == (4, 4)
        assert loaded.training_mode == "test"
        assert loaded.used_device == "cpu"
        assert loaded.used_seed == 42


def test_saving_without_mapf():
    stats = TrainingStats(training_mode="lacam_only")
    stats.record_epoch(loss=0.3)
    stats.record_epoch(loss=0.2)

    with tempfile.TemporaryDirectory() as tmp:
        stats.save(tmp)

        loaded = TrainingStats.load(tmp)

        assert loaded._epoch_count == 2
        assert loaded._losses == [0.3, 0.2]
        assert loaded.mapf is None
