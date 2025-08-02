"""Tests for data preparation module."""
import pytest
from pathlib import Path

from src.data.prepare_data import process_raw_data, create_synthetic_data

def test_data_splitting():
    """Test data splitting ratios."""
    train_ds, val_ds, test_ds = process_raw_data(test_size=0.2, val_size=0.1)
    
    total_size = len(train_ds) + len(val_ds) + len(test_ds)
    
    # Check approximate split ratios (allowing for rounding)
    assert abs(len(train_ds) / total_size - 0.7) < 0.02
    assert abs(len(val_ds) / total_size - 0.1) < 0.02
    assert abs(len(test_ds) / total_size - 0.2) < 0.02

def test_synthetic_data_generation():
    """Test synthetic data generation."""
    synthetic_data = create_synthetic_data(num_samples=100)
    
    assert len(synthetic_data) == 100
    for item in synthetic_data:
        assert "text" in item
        assert "label" in item
        assert isinstance(item["label"], int)
        assert item["label"] in [0, 1]
