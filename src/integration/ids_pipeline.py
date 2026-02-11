import os
import numpy as np
import time
import logging

from src.tier1_signature.signature_detector import Tier1SignatureDetector
from src.integration.alert_manager import AlertManager
from src.utils.config import resolve_path

logger = logging.getLogger('IDS')


class AdversarialRobustIDS:
    """
    Main IDS Pipeline: Three-tier intrusion detection.

    Flow:
    1. Traffic -> Tier 1 (signature match?) -> YES: Alert | NO: Continue
    2. Traffic -> Tier 2 (ML detects anomaly?) -> NO: Benign | YES: Continue
    3. Traffic -> Tier 3 (adversarial check?) -> Alert with adversarial flag
    """

    def __init__(self, config):
        self.config = config
        self.tier1 = None
        self.tier2 = None
        self.tier3 = None
        self.alert_manager = AlertManager()
        self.stats = {'total': 0, 'tier1': 0, 'tier2': 0, 'tier3': 0, 'benign': 0}

        self._init_tier1()
        self._init_tier2()
        self._init_tier3()

    def _init_tier1(self):
        """Initialize Tier 1 signature detector."""
        if not self.config['tier1']['enabled']:
            return
        try:
            sig_path = resolve_path(self.config['tier1']['signature_db'])
            self.tier1 = Tier1SignatureDetector(sig_path)
            logger.info("Tier 1 (Signature Detection) initialized")
        except Exception as e:
            logger.warning(f"Tier 1 initialization failed: {e}")

    def _init_tier2(self):
        """Initialize Tier 2 ML detector."""
        if not self.config['tier2']['enabled']:
            return
        try:
            model_path = resolve_path(self.config['tier2']['model_path'])
            if os.path.exists(model_path):
                from src.tier2_ml_detection.ml_detector import Tier2MLDetector
                self.tier2 = Tier2MLDetector(
                    model_path=model_path,
                    model_type=self.config['tier2']['model_type'],
                    confidence_threshold=self.config['tier2']['confidence_threshold']
                )
                logger.info("Tier 2 (ML Detection) initialized")
            else:
                logger.warning(f"Tier 2 model not found at {model_path}. Tier 2 disabled.")
        except Exception as e:
            logger.warning(f"Tier 2 initialization failed: {e}")

    def _init_tier3(self):
        """Initialize Tier 3 adversarial defense."""
        if not self.config['tier3']['enabled']:
            return
        try:
            robust_path = resolve_path(self.config['tier3']['robust_model_path'])
            if os.path.exists(robust_path):
                import torch
                from src.adversarial_attacks.attack_utils import PyTorchDNN
                from src.tier3_adversarial_defense.adversarial_defense import Tier3AdversarialDefense
                from src.tier3_adversarial_defense.ensemble_defense import ModelWrapper

                # Load robust model - we need to know dimensions
                # Try loading the state dict and infer dimensions
                state = torch.load(robust_path, map_location='cpu')
                first_key = list(state.keys())[0]
                input_dim = state[first_key].shape[1]
                last_key = list(state.keys())[-1]
                num_classes = state[last_key].shape[0]

                robust_model = PyTorchDNN(input_dim, num_classes)
                robust_model.load_state_dict(state)
                robust_model.eval()

                robust_wrapper = ModelWrapper(robust_model, framework='pytorch')

                # For ensemble, use the robust model as the only member if others not available
                ensemble_models = [robust_wrapper]

                self.tier3 = Tier3AdversarialDefense(
                    robust_model=robust_wrapper,
                    ensemble_models=ensemble_models,
                    detection_threshold=self.config['tier3']['adversarial_detection_threshold']
                )
                logger.info("Tier 3 (Adversarial Defense) initialized")
            else:
                logger.warning(f"Tier 3 model not found at {robust_path}. Tier 3 disabled.")
        except Exception as e:
            logger.warning(f"Tier 3 initialization failed: {e}")

    def detect(self, traffic_sample):
        """
        Run full three-tier detection on a single sample.

        Args:
            traffic_sample: dict (raw features for Tier 1)
                            OR numpy array (preprocessed for Tier 2/3)
        """
        start_time = time.time()
        self.stats['total'] += 1

        # === TIER 1: Signature Detection ===
        if self.tier1 is not None and isinstance(traffic_sample, dict):
            tier1_result = self.tier1.detect(traffic_sample)

            if tier1_result['is_attack']:
                self.stats['tier1'] += 1
                return self.alert_manager.create_alert(
                    tier=1,
                    attack_type=tier1_result['attack_type'],
                    severity=tier1_result['severity'],
                    confidence=1.0,
                    is_adversarial=False,
                    total_time_ms=(time.time() - start_time) * 1000
                )

        # === TIER 2: ML Detection ===
        if self.tier2 is not None:
            features = traffic_sample if isinstance(traffic_sample, np.ndarray) else None
            if features is None:
                # Cannot run Tier 2 without preprocessed features
                self.stats['benign'] += 1
                return {
                    'status': 'BENIGN',
                    'tier': 1,
                    'note': 'No ML features available',
                    'total_time_ms': (time.time() - start_time) * 1000
                }

            tier2_result = self.tier2.detect(features)

            if not tier2_result['is_attack']:
                self.stats['benign'] += 1
                return {
                    'status': 'BENIGN',
                    'tier': 2,
                    'confidence': tier2_result['confidence'],
                    'total_time_ms': (time.time() - start_time) * 1000
                }

            # === TIER 3: Adversarial Robustness Check ===
            if self.tier3 is not None:
                tier3_result = self.tier3.detect_and_classify(features)

                if tier3_result['is_adversarial']:
                    self.stats['tier3'] += 1
                    return self.alert_manager.create_alert(
                        tier=3,
                        attack_type=tier2_result['attack_type'],
                        severity='CRITICAL',
                        confidence=tier3_result['confidence'],
                        is_adversarial=True,
                        adversarial_score=tier3_result['adversarial_score'],
                        total_time_ms=(time.time() - start_time) * 1000
                    )

            # Normal ML detection (not adversarial)
            self.stats['tier2'] += 1
            return self.alert_manager.create_alert(
                tier=2,
                attack_type=tier2_result['attack_type'],
                severity='HIGH' if tier2_result['confidence'] > 0.9 else 'MEDIUM',
                confidence=tier2_result['confidence'],
                is_adversarial=False,
                total_time_ms=(time.time() - start_time) * 1000
            )

        # No ML tier available
        self.stats['benign'] += 1
        return {
            'status': 'BENIGN',
            'tier': 1,
            'note': 'Only Tier 1 available',
            'total_time_ms': (time.time() - start_time) * 1000
        }

    def detect_batch(self, samples):
        """Run detection on a batch of samples."""
        return [self.detect(s) for s in samples]

    def get_statistics(self):
        """Return detection statistics."""
        return {
            'total_traffic': self.stats['total'],
            'tier1_detections': self.stats['tier1'],
            'tier2_detections': self.stats['tier2'],
            'tier3_detections': self.stats['tier3'],
            'benign': self.stats['benign'],
            'alerts': self.alert_manager.get_alert_summary()
        }
