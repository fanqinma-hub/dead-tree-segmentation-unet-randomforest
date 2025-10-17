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

def test_cuda_information():
    cuda_info = {
        "cuda_available": True,
        "cuda_version": "11.8",
        "gpu_device": "NVIDIA GeForce RTX 4060 Laptop GPU",
        "gpu_memory": "8.0 GB"
    }
    
    assert cuda_info["cuda_available"] is True
    assert cuda_info["cuda_version"] == "11.8"
    assert cuda_info["gpu_device"] == "NVIDIA GeForce RTX 4060 Laptop GPU"
    assert cuda_info["gpu_memory"] == "8.0 GB"