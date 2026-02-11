import pytest
import numpy as np
import torch
from src.tier3_adversarial_defense.input_transformation import (
    bit_depth_reduction, gaussian_smoothing, feature_squeezing
)
from src.tier3_adversarial_defense.ensemble_defense import EnsembleDefense, ModelWrapper
from src.adversarial_attacks.attack_utils import PyTorchDNN


def test_bit_depth_reduction_range():
    x = np.random.rand(10, 35).astype(np.float32)
    x_reduced = bit_depth_reduction(x, depth=4)
    assert x_reduced.min() >= 0
    assert x_reduced.max() <= 1
    assert x_reduced.shape == x.shape


def test_gaussian_smoothing_shape():
    x = np.random.rand(10, 35).astype(np.float32)
    x_smoothed = gaussian_smoothing(x, sigma=0.1)
    assert x_smoothed.shape == x.shape
    assert x_smoothed.min() >= 0
    assert x_smoothed.max() <= 1


def test_feature_squeezing():
    x = np.random.rand(5, 35).astype(np.float32)
    x_squeezed = feature_squeezing(x, bit_depth=4)
    assert x_squeezed.shape == x.shape


def test_ensemble_defense_single_model():
    model = PyTorchDNN(35, 5)
    wrapper = ModelWrapper(model, framework='pytorch')
    ensemble = EnsembleDefense([wrapper])

    x = np.random.rand(10, 35).astype(np.float32)
    result = ensemble.predict(x)

    assert 'predictions' in result
    assert 'confidence' in result
    assert 'agreement' in result
    assert len(result['predictions']) == 10


def test_ensemble_defense_multiple_models():
    model1 = PyTorchDNN(35, 5)
    model2 = PyTorchDNN(35, 5)
    wrapper1 = ModelWrapper(model1, framework='pytorch')
    wrapper2 = ModelWrapper(model2, framework='pytorch')
    ensemble = EnsembleDefense([wrapper1, wrapper2])

    x = np.random.rand(5, 35).astype(np.float32)
    result = ensemble.predict(x)

    assert len(result['predictions']) == 5
    assert all(0 <= c <= 1 for c in result['confidence'])
    assert all(0 <= a <= 1 for a in result['agreement'])


def test_adversarial_defense_detect():
    from src.tier3_adversarial_defense.adversarial_defense import Tier3AdversarialDefense

    model = PyTorchDNN(35, 5)
    wrapper = ModelWrapper(model, framework='pytorch')

    defense = Tier3AdversarialDefense(
        robust_model=wrapper,
        ensemble_models=[wrapper],
        detection_threshold=0.1
    )

    x = np.random.rand(35).astype(np.float32)
    result = defense.detect_and_classify(x)

    assert 'is_adversarial' in result
    assert 'adversarial_score' in result
    assert 'prediction' in result
    assert 'confidence' in result
    assert 'tier' in result
    assert result['tier'] == 3
