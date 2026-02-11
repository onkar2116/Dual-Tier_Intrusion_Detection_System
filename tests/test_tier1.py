import pytest
from src.tier1_signature.signature_detector import Tier1SignatureDetector


@pytest.fixture
def detector(signature_db_path):
    return Tier1SignatureDetector(signature_db_path)


def test_detects_dos_attack(detector):
    sample = {
        'protocol': 'TCP',
        'syn_flag_count': 200,
        'ack_flag_count': 2,
        'flow_duration': 500
    }
    result = detector.detect(sample)
    assert result['is_attack'] is True
    assert result['attack_type'] == 'SYN Flood'
    assert result['tier'] == 1


def test_benign_traffic_passes(detector):
    sample = {
        'protocol': 'TCP',
        'syn_flag_count': 1,
        'ack_flag_count': 1,
        'flow_duration': 5000
    }
    result = detector.detect(sample)
    assert result['is_attack'] is False


def test_detection_speed(detector):
    sample = {'protocol': 'TCP', 'syn_flag_count': 5, 'ack_flag_count': 5}
    result = detector.detect(sample)
    assert result['detection_time_ms'] < 100  # Should be very fast


def test_batch_detection(detector):
    samples = [
        {'protocol': 'TCP', 'syn_flag_count': 200, 'ack_flag_count': 2, 'flow_duration': 500},
        {'protocol': 'TCP', 'syn_flag_count': 1, 'ack_flag_count': 1, 'flow_duration': 5000},
    ]
    results = detector.detect_batch(samples)
    assert len(results) == 2
    assert results[0]['is_attack'] is True
    assert results[1]['is_attack'] is False


def test_nsl_kdd_signature_detection(detector):
    """Test NSL-KDD specific signatures."""
    sample = {
        'protocol_type': 'icmp',
        'src_bytes': 1000,
        'dst_bytes': 0
    }
    result = detector.detect(sample)
    assert result['is_attack'] is True
    assert 'Smurf' in result.get('attack_type', '')
