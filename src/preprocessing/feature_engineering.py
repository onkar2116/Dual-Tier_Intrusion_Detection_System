import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, chi2
from sklearn.ensemble import RandomForestClassifier


def compute_feature_importance(X, y, feature_names, method='mutual_info'):
    """
    Compute feature importance scores.

    Args:
        X: Feature matrix (numpy array)
        y: Labels (numpy array)
        feature_names: List of feature names
        method: 'mutual_info', 'chi2', or 'random_forest'

    Returns:
        DataFrame with feature names and importance scores, sorted descending.
    """
    if method == 'mutual_info':
        scores = mutual_info_classif(X, y, random_state=42)
    elif method == 'chi2':
        X_positive = np.abs(X)
        scores, _ = chi2(X_positive, y)
    elif method == 'random_forest':
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)
        scores = rf.feature_importances_
    else:
        raise ValueError(f"Unknown method: {method}")

    importance_df = pd.DataFrame({
        'feature': feature_names[:len(scores)],
        'importance': scores
    }).sort_values('importance', ascending=False).reset_index(drop=True)

    return importance_df


def get_top_features(X, y, feature_names, k=35, method='mutual_info'):
    """Select top-k features and return their names and indices."""
    importance_df = compute_feature_importance(X, y, feature_names, method)
    top_features = importance_df.head(k)['feature'].tolist()
    top_indices = [feature_names.index(f) for f in top_features if f in feature_names]
    return top_features, top_indices


def get_correlation_matrix(X, feature_names):
    """Compute correlation matrix for feature analysis."""
    df = pd.DataFrame(X, columns=feature_names[:X.shape[1]])
    return df.corr()
