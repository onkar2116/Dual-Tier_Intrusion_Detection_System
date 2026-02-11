import pytest
import numpy as np


def test_dnn_model_builds():
    from src.tier2_ml_detection.models import build_dnn
    model = build_dnn(input_dim=35, num_classes=5)
    assert model is not None
    assert model.input_shape == (None, 35)


def test_dnn_model_predicts():
    from src.tier2_ml_detection.models import build_dnn
    model = build_dnn(input_dim=35, num_classes=5)
    x = np.random.randn(10, 35).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    assert predictions.shape == (10, 5)
    # Probabilities should sum to ~1
    assert np.allclose(predictions.sum(axis=1), 1.0, atol=0.01)


def test_cnn_model_builds():
    from src.tier2_ml_detection.models import build_cnn
    model = build_cnn(input_shape=(35, 1), num_classes=5)
    assert model is not None


def test_cnn_model_predicts():
    from src.tier2_ml_detection.models import build_cnn
    model = build_cnn(input_shape=(35, 1), num_classes=5)
    x = np.random.randn(10, 35, 1).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    assert predictions.shape == (10, 5)


def test_lstm_model_builds():
    from src.tier2_ml_detection.models import build_lstm
    model = build_lstm(input_shape=(1, 35), num_classes=5)
    assert model is not None


def test_lstm_model_predicts():
    from src.tier2_ml_detection.models import build_lstm
    model = build_lstm(input_shape=(1, 35), num_classes=5)
    x = np.random.randn(10, 1, 35).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    assert predictions.shape == (10, 5)


def test_binary_model():
    from src.tier2_ml_detection.models import build_dnn
    model = build_dnn(input_dim=20, num_classes=2)
    x = np.random.randn(5, 20).astype(np.float32)
    predictions = model.predict(x, verbose=0)
    assert predictions.shape[0] == 5


def test_feature_extractor():
    from src.tier2_ml_detection.feature_extractor import FeatureExtractor

    ext_dnn = FeatureExtractor('DNN', 35)
    ext_cnn = FeatureExtractor('CNN', 35)
    ext_lstm = FeatureExtractor('LSTM', 35)

    x = np.random.randn(10, 35).astype(np.float32)

    assert ext_dnn.reshape(x).shape == (10, 35)
    assert ext_cnn.reshape(x).shape == (10, 35, 1)
    assert ext_lstm.reshape(x).shape == (10, 1, 35)
