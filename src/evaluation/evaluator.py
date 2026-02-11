import numpy as np
import os
import json
from tqdm import tqdm

from src.evaluation.metrics import compute_all_metrics, compute_robust_accuracy
from src.utils.config import resolve_path


class SystemEvaluator:
    """Comprehensive evaluation framework for the three-tier IDS."""

    def __init__(self, config):
        self.config = config
        self.results = {}

    def evaluate_clean(self, model, X_test, y_test, model_name='model'):
        """Evaluate model on clean (unperturbed) test data."""
        predictions = model.predict(X_test, verbose=0) if hasattr(model, 'predict') else model(X_test)

        if hasattr(predictions, 'numpy'):
            predictions = predictions.numpy()
        predictions = np.array(predictions)

        if predictions.ndim > 1:
            y_pred = np.argmax(predictions, axis=1)
            y_probs = predictions
        else:
            y_pred = (predictions > 0.5).astype(int)
            y_probs = None

        metrics = compute_all_metrics(y_test, y_pred, y_probs)
        self.results[f'{model_name}_clean'] = metrics
        return metrics

    def evaluate_robust(self, model_predict_fn, X_test, y_test, X_adv, attack_name):
        """Evaluate model robustness against adversarial attack."""
        pred_clean = model_predict_fn(X_test)
        pred_adv = model_predict_fn(X_adv)

        if hasattr(pred_clean, 'numpy'):
            pred_clean = pred_clean.numpy()
        if hasattr(pred_adv, 'numpy'):
            pred_adv = pred_adv.numpy()

        pred_clean = np.array(pred_clean)
        pred_adv = np.array(pred_adv)

        if pred_clean.ndim > 1:
            y_pred_clean = np.argmax(pred_clean, axis=1)
            y_pred_adv = np.argmax(pred_adv, axis=1)
        else:
            y_pred_clean = (pred_clean > 0.5).astype(int)
            y_pred_adv = (pred_adv > 0.5).astype(int)

        robust_metrics = compute_robust_accuracy(y_test, y_pred_clean, y_pred_adv)
        self.results[f'robust_{attack_name}'] = robust_metrics
        return robust_metrics

    def compare_baselines(self, baseline_results):
        """
        Compare different system configurations.

        Args:
            baseline_results: dict of {system_name: metrics_dict}
        """
        comparison = []
        for name, metrics in baseline_results.items():
            comparison.append({
                'System': name,
                'Accuracy': metrics.get('accuracy', 0),
                'Precision': metrics.get('precision', 0),
                'Recall': metrics.get('recall', 0),
                'F1': metrics.get('f1_score', 0),
                'FPR': metrics.get('fpr', 0),
            })
        self.results['baseline_comparison'] = comparison
        return comparison

    def run_full_evaluation(self, model, X_test, y_test, model_name='model'):
        """Run complete evaluation suite."""
        print(f"\n{'='*60}")
        print(f"Running Full Evaluation for {model_name}")
        print(f"{'='*60}")

        # Clean evaluation
        clean_metrics = self.evaluate_clean(model, X_test, y_test, model_name)
        print(f"\nClean Accuracy: {clean_metrics['accuracy']:.4f}")
        print(f"Precision: {clean_metrics['precision']:.4f}")
        print(f"Recall: {clean_metrics['recall']:.4f}")
        print(f"F1-Score: {clean_metrics['f1_score']:.4f}")
        print(f"FPR: {clean_metrics.get('fpr', 'N/A')}")

        return self.results

    def save_results(self, output_dir='evaluation_results'):
        """Save all evaluation results to JSON."""
        save_dir = resolve_path(output_dir)
        os.makedirs(save_dir, exist_ok=True)

        # Convert numpy types to Python types for JSON serialization
        serializable = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                serializable[key] = {
                    k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                    for k, v in value.items()
                }
            else:
                serializable[key] = value

        with open(os.path.join(save_dir, 'evaluation_results.json'), 'w') as f:
            json.dump(serializable, f, indent=2, default=str)

        print(f"Results saved to {save_dir}")
