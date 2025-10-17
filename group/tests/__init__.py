import pytest

def test_training_configuration():
    config = {
        "batch_size": 8,
        "num_workers": 2,
        "learning_rate": 0.0001,
        "num_epochs": 100,
        "device": "cuda"
    }
    assert config["batch_size"] == 8
    assert config["num_workers"] == 2
    assert config["learning_rate"] == 0.0001
    assert config["num_epochs"] == 100
    assert config["device"] == "cuda"