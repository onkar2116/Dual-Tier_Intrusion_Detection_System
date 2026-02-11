import pytest
import torch
import numpy as np
from src.adversarial_attacks.fgsm import fgsm_attack
from src.adversarial_attacks.pgd import pgd_attack
from src.adversarial_attacks.attack_utils import PyTorchDNN, evaluate_attack


def create_simple_model(input_dim=10, num_classes=2):
    return PyTorchDNN(input_dim, num_classes)


def test_fgsm_perturbation_bounded():
    model = create_simple_model()
    x = torch.rand(5, 10)
    y = torch.zeros(5, dtype=torch.long)
    epsilon = 0.1

    x_adv = fgsm_attack(model, x, y, epsilon=epsilon)

    perturbation = (x_adv - x).abs()
    assert perturbation.max() <= epsilon + 1e-5


def test_pgd_perturbation_bounded():
    model = create_simple_model()
    x = torch.rand(5, 10)
    y = torch.zeros(5, dtype=torch.long)
    epsilon = 0.1

    x_adv = pgd_attack(model, x, y, epsilon=epsilon, alpha=0.01, num_iterations=10)

    perturbation = (x_adv - x).abs()
    assert perturbation.max() <= epsilon + 1e-5


def test_adversarial_output_valid_range():
    model = create_simple_model()
    x = torch.rand(5, 10)
    y = torch.zeros(5, dtype=torch.long)

    x_adv = fgsm_attack(model, x, y, epsilon=0.1)

    assert x_adv.min() >= 0
    assert x_adv.max() <= 1


def test_evaluate_attack_function():
    model = create_simple_model()
    x_clean = np.random.rand(20, 10).astype(np.float32)
    y_true = np.zeros(20, dtype=np.int64)

    x_adv = fgsm_attack(model, torch.FloatTensor(x_clean), torch.LongTensor(y_true), 0.1).numpy()

    result = evaluate_attack(model, x_clean, x_adv, y_true)
    assert 'attack_success_rate' in result
    assert 'avg_l2_perturbation' in result
    assert result['avg_l2_perturbation'] >= 0


def test_pytorch_dnn_forward():
    model = PyTorchDNN(35, 5)
    x = torch.randn(10, 35)
    output = model(x)
    assert output.shape == (10, 5)
