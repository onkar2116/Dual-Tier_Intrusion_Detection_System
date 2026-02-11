import pytest
import numpy as np
from src.integration.alert_manager import AlertManager


def test_alert_creation():
    manager = AlertManager.__new__(AlertManager)
    manager.alerts = []
    manager.logger = __import__('logging').getLogger('test')

    alert = manager.create_alert(
        tier=1,
        attack_type='SYN Flood',
        severity='HIGH',
        confidence=1.0,
        is_adversarial=False
    )

    assert 'alert_id' in alert
    assert alert['tier'] == 1
    assert alert['attack_type'] == 'SYN Flood'
    assert alert['severity'] == 'HIGH'
    assert alert['is_adversarial'] is False


def test_alert_priority():
    manager = AlertManager.__new__(AlertManager)
    manager.alerts = []
    manager.logger = __import__('logging').getLogger('test')

    # Adversarial should be priority 1
    assert manager._calculate_priority(3, 'CRITICAL', True) == 1

    # Tier 1 CRITICAL should be priority 1
    assert manager._calculate_priority(1, 'CRITICAL', False) == 1

    # HIGH should be priority 2
    assert manager._calculate_priority(2, 'HIGH', False) == 2

    # MEDIUM should be priority 3
    assert manager._calculate_priority(2, 'MEDIUM', False) == 3


def test_alert_summary():
    manager = AlertManager.__new__(AlertManager)
    manager.alerts = []
    manager.logger = __import__('logging').getLogger('test')

    manager.create_alert(tier=1, attack_type='A', severity='HIGH', confidence=1.0)
    manager.create_alert(tier=2, attack_type='B', severity='MEDIUM', confidence=0.8)
    manager.create_alert(tier=3, attack_type='C', severity='CRITICAL', confidence=0.9, is_adversarial=True)

    summary = manager.get_alert_summary()
    assert summary['total_alerts'] == 3
    assert summary['by_tier'][1] == 1
    assert summary['by_tier'][2] == 1
    assert summary['by_tier'][3] == 1
    assert summary['adversarial_count'] == 1


def test_tier1_in_pipeline(config, signature_db_path):
    """Test Tier 1 detection through pipeline config."""
    from src.tier1_signature.signature_detector import Tier1SignatureDetector

    detector = Tier1SignatureDetector(signature_db_path)

    # Known attack
    attack = {
        'protocol': 'TCP', 'syn_flag_count': 200,
        'ack_flag_count': 2, 'flow_duration': 500
    }
    result = detector.detect(attack)
    assert result['is_attack'] is True
    assert result['tier'] == 1

    # Benign
    benign = {
        'protocol': 'TCP', 'syn_flag_count': 1,
        'ack_flag_count': 1, 'flow_duration': 5000
    }
    result = detector.detect(benign)
    assert result['is_attack'] is False
