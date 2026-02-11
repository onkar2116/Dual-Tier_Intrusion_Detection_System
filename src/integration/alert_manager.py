from datetime import datetime
import uuid
import json
import logging
import os

from src.utils.config import resolve_path


class AlertManager:
    """Manages alert creation, prioritization, and logging."""

    def __init__(self, log_file='logs/alerts.log'):
        self.alerts = []
        self.logger = logging.getLogger('AlertManager')

        if not self.logger.handlers:
            log_path = resolve_path(log_file)
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            handler = logging.FileHandler(log_path)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)

    def create_alert(self, tier, attack_type, severity, confidence,
                     is_adversarial=False, **kwargs):
        """Create and log a detection alert."""
        alert = {
            'alert_id': str(uuid.uuid4())[:8],
            'timestamp': datetime.now().isoformat(),
            'tier': tier,
            'attack_type': attack_type,
            'severity': severity,
            'confidence': round(confidence, 4),
            'is_adversarial': is_adversarial,
            'priority': self._calculate_priority(tier, severity, is_adversarial),
            **kwargs
        }

        self.alerts.append(alert)
        self.logger.info(json.dumps(alert))
        return alert

    def _calculate_priority(self, tier, severity, is_adversarial):
        """Assign priority score (1=highest, 5=lowest)."""
        if is_adversarial:
            return 1
        if tier == 1 and severity == 'CRITICAL':
            return 1
        if severity == 'HIGH':
            return 2
        if severity == 'MEDIUM':
            return 3
        return 4

    def get_recent_alerts(self, n=50):
        return self.alerts[-n:]

    def get_alert_summary(self):
        return {
            'total_alerts': len(self.alerts),
            'by_tier': {
                1: sum(1 for a in self.alerts if a['tier'] == 1),
                2: sum(1 for a in self.alerts if a['tier'] == 2),
                3: sum(1 for a in self.alerts if a['tier'] == 3),
            },
            'adversarial_count': sum(1 for a in self.alerts if a['is_adversarial'])
        }
