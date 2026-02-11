import pytest
import numpy as np
from src.preprocessing.data_loader import DataLoader
from src.preprocessing.preprocessor import DataPreprocessor


def test_synthetic_data_generation():
    loader = DataLoader()
    df = loader.generate_synthetic(n_samples=100)
    assert len(df) == 100
    assert 'label' in df.columns
    assert 'attack_category' in df.columns
    assert 'protocol_type' in df.columns


def test_synthetic_data_label_distribution():
    loader = DataLoader()
    df = loader.generate_synthetic(n_samples=1000, random_seed=42)
    # Should have Normal and attack categories
    categories = df['attack_category'].unique()
    assert 'Normal' in categories
    assert len(categories) >= 3  # At least Normal + 2 attack types


def test_preprocessing_pipeline(config, synthetic_data):
    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(synthetic_data, label_type='multiclass')

    assert 'X_train' in data
    assert 'X_val' in data
    assert 'X_test' in data
    assert 'y_train' in data
    assert data['X_train'].shape[1] == data['X_val'].shape[1] == data['X_test'].shape[1]
    assert data['n_features'] > 0
    assert data['n_classes'] >= 2


def test_preprocessing_binary_labels(config, synthetic_data):
    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(synthetic_data, label_type='binary')

    unique_labels = np.unique(data['y_test'])
    assert set(unique_labels).issubset({0, 1})


def test_scaling_consistency(config, synthetic_data):
    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(synthetic_data, label_type='multiclass')

    # Check scaled data has reasonable range (standard scaling: mean~0, std~1)
    train_mean = np.mean(data['X_train'], axis=0)
    # After SMOTE the mean might shift, but should be roughly centered
    assert np.all(np.abs(train_mean) < 5)


def test_data_types(config, synthetic_data):
    preprocessor = DataPreprocessor(config)
    data = preprocessor.run_pipeline(synthetic_data, label_type='multiclass')

    assert data['X_train'].dtype == np.float32
    assert data['y_train'].dtype == np.int64
