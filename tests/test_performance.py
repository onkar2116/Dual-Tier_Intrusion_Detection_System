import pytest
import time
import numpy as np


def test_tier1_throughput(signature_db_path):
    """Tier 1 should process > 100 samples/sec."""
    from src.tier1_signature.signature_detector import Tier1SignatureDetector

    detector = Tier1SignatureDetector(signature_db_path)
    samples = [{'protocol': 'TCP', 'syn_flag_count': i % 10, 'ack_flag_count': 5}
               for i in range(100)]

    start = time.time()
    for s in samples:
        detector.detect(s)
    elapsed = time.time() - start

    throughput = 100 / elapsed
    assert throughput > 100, f"Tier 1 throughput too low: {throughput:.0f} samples/sec"


def test_dnn_inference_time():
    """DNN inference should be < 100ms per sample."""
    from src.tier2_ml_detection.models import build_dnn

    model = build_dnn(input_dim=35, num_classes=5)
    x = np.random.randn(1, 35).astype(np.float32)

    # Warm up
    model.predict(x, verbose=0)

    start = time.time()
    for _ in range(10):
        model.predict(x, verbose=0)
    elapsed = (time.time() - start) / 10 * 1000

    assert elapsed < 100, f"DNN inference too slow: {elapsed:.1f}ms per sample"


def test_fgsm_attack_speed():
    """FGSM attack should be < 500ms for small batch."""
    import torch
    from src.adversarial_attacks.attack_utils import PyTorchDNN
    from src.adversarial_attacks.fgsm import fgsm_attack

    model = PyTorchDNN(35, 5)
    x = torch.rand(10, 35)
    y = torch.randint(0, 5, (10,))

    start = time.time()
    fgsm_attack(model, x, y, epsilon=0.1)
    elapsed = (time.time() - start) * 1000

    assert elapsed < 500, f"FGSM too slow: {elapsed:.1f}ms"
