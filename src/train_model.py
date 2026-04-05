"""
Trains two models (Random Forest + XGBoost), compares them,
saves the best one, and prints feature importances.
"""

import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.ensemble         import RandomForestRegressor
from sklearn.metrics          import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing    import StandardScaler
from xgboost                  import XGBRegressor

# ── All feature columns used for training ────────────────────────────────────
# These are the output columns from features.py (Components 1, 2, 3)
FEATURE_COLS = [
    # Component 1 — Text Structure
    'vader_compound', 'vader_positive', 'vader_negative', 'vader_neutral',
    'sentiment_agreement', 'flesch_ease', 'flesch_grade',
    'word_count', 'char_count', 'sentence_count',
    'avg_word_length', 'avg_sentence_length',
    'feature_mentions', 'pros_count', 'cons_count', 'has_both_sides',

    # Component 2 — Grammar Structure
    'noun_ratio', 'verb_ratio', 'adj_ratio', 'adv_ratio', 'pos_diversity',
    'emotion_count', 'emotion_ratio',
    'all_caps_ratio', 'exclaim_count', 'exclaim_ratio',
    'question_count', 'question_ratio',
    'lexical_diversity', 'paragraph_count', 'number_count', 'has_numbers',

    # Component 3 — User Activity Signals
    'score', 'user_review_count', 'user_avg_score', 'user_score_std',
    'reviews_per_day', 'is_suspicious',
    'product_avg_score', 'product_review_count', 'score_deviation',
]

TARGET_COL = 'helpfulness_score'


# ── Load & validate data ──────────────────────────────────────────────────────
def load_features(path: str = "data/reviews_features.csv"):
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")

    # Check all feature columns exist
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()

    # Fill any remaining NaN (e.g. users with 1 review → std is NaN)
    X.fillna(0, inplace=True)

    print(f"X shape : {X.shape}")
    print(f"y range : {y.min():.3f} – {y.max():.3f}  |  mean: {y.mean():.3f}")
    return X, y


# ── Train / evaluate ──────────────────────────────────────────────────────────
def evaluate(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse   = np.sqrt(mean_squared_error(y_test, y_pred))
    mae    = mean_absolute_error(y_test, y_pred)
    r2     = r2_score(y_test, y_pred)
    print(f"\n  [{name}]")
    print(f"    RMSE : {rmse:.4f}  (lower is better)")
    print(f"    MAE  : {mae:.4f}  (lower is better)")
    print(f"    R²   : {r2:.4f}  (closer to 1 is better)")
    return {'name': name, 'model': model, 'rmse': rmse, 'mae': mae, 'r2': r2}


def train(data_path: str = "reviews_features.csv"):
    X, y = load_features(data_path)

    # 80/20 split — stratified by score bucket for balanced test set
    strata = pd.cut(y, bins=[0, 0.4, 0.75, 1.01], labels=[0, 1, 2], include_lowest=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=strata
    )
    print(f"\nTrain: {len(X_train)} rows  |  Test: {len(X_test)} rows")

    results = []

    # ── Model 1: Random Forest ────────────────────────────────────────────
    print("\nTraining Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=200,   # 200 trees
        max_depth=12,       # prevent overfitting
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1           # use all CPU cores
    )
    rf.fit(X_train, y_train)
    results.append(evaluate("Random Forest", rf, X_test, y_test))

    # ── Model 2: XGBoost ──────────────────────────────────────────────────
    print("\nTraining XGBoost...")
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    xgb.fit(X_train, y_train)
    results.append(evaluate("XGBoost", xgb, X_test, y_test))

    # ── Pick best model (lowest RMSE) ────────────────────────────────────
    best = min(results, key=lambda r: r['rmse'])
    print(f"\nBest model: {best['name']}  (RMSE={best['rmse']:.4f})")

    # ── Save best model + feature list ───────────────────────────────────
    os.makedirs("models", exist_ok=True)
    joblib.dump(best['model'],  "models/helpfulness_model.pkl")
    joblib.dump(FEATURE_COLS,   "models/feature_cols.pkl")
    print("Model saved → models/helpfulness_model.pkl")

    # ── Feature importances (top 15) ─────────────────────────────────────
    importances = pd.Series(
        best['model'].feature_importances_,
        index=FEATURE_COLS
    ).sort_values(ascending=False).head(15)

    print("\nTop 15 features by importance:")
    print(importances.round(4).to_string())

    plt.figure(figsize=(10, 6))
    importances.sort_values().plot(kind='barh', color='steelblue', edgecolor='white')
    plt.title(f"Top 15 Feature Importances — {best['name']}")
    plt.xlabel("Importance score")
    plt.tight_layout()
    plt.savefig("models/feature_importance.png", dpi=120)
    plt.show()
    print("Chart saved → models/feature_importance.png")

    return best['model']


if __name__ == "__main__":
    train()