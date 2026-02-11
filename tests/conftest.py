import os
import sys
import pytest
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import load_config
from src.preprocessing.data_loader import DataLoader


@pytest.fixture
def config():
    """Load project configuration."""
    return load_config()


@pytest.fixture
def synthetic_data():
    """Generate synthetic data for testing."""
    loader = DataLoader()
    return loader.generate_synthetic(n_samples=200, random_seed=42)


@pytest.fixture
def preprocessed_data(config, synthetic_data):
    """Preprocessed synthetic data."""
    from src.preprocessing.preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor(config)
    return preprocessor.run_pipeline(synthetic_data, label_type='multiclass')


@pytest.fixture
def signature_db_path(config):
    """Path to signature database."""
    from src.utils.config import resolve_path
    return resolve_path(config['tier1']['signature_db'])
