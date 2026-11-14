"""Baseline vs. Random Forest on a real health dataset from Hugging Face.

Usage:
    python baseline_test.py            # UCI Heart Disease (default)
    python baseline_test.py diabetes   # Pima Indians Diabetes

Datasets (cite these in your write-up):
  heart    - UCI Heart Disease (Cleveland), 303 patients
             https://huggingface.co/datasets/skrishna/heart_disease_uci
             original: https://archive.ics.uci.edu/dataset/45/heart+disease
  diabetes - Pima Indians Diabetes Database, 768 patients
             https://huggingface.co/datasets/khoaguin/pima-indians-diabetes-database

Prototype for learning only - not medical advice.
"""

import sys

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
HF = "https://huggingface.co/datasets"

DATASETS = {
    "heart": {
        "url": f"{HF}/skrishna/heart_disease_uci/resolve/main/test.csv",
        "target": "target",  # 1 = heart disease (>50% narrowing)
        "positive": "heart disease",
        # In this data, 0 is a real value for every column, so nothing to fix.
        "zero_means_missing": [],
        "group_col": "sex",  # 1 = male, 0 = female
        "group_names": {0: "female", 1: "male"},
    },
    "diabetes": {
        "url": f"{HF}/khoaguin/pima-indians-diabetes-database/resolve/main/diabetes.csv",
        "target": "y",  # 1 = diabetes
        "positive": "diabetes",
        # A glucose or BMI of 0 is impossible: these zeros are missing values.
        "zero_means_missing": ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"],
        "group_col": None,  # no sex/ethnicity column (all patients are Pima women)
        "group_names": {},
    },
}


def load(name: str) -> tuple[pd.DataFrame, pd.Series, dict]:
    cfg = DATASETS[name]
    df = pd.read_csv(cfg["url"])
    df = df.replace("?", np.nan)  # some UCI copies mark missing values with "?"
    for col in cfg["zero_means_missing"]:
        df[col] = df[col].replace(0, np.nan)
    y = df.pop(cfg["target"]).astype(int)
    X = df.apply(pd.to_numeric, errors="coerce")
    return X, y, cfg


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "heart"
    if name not in DATASETS:
        sys.exit(f"Unknown dataset '{name}'. Choose one of: {', '.join(DATASETS)}")

    X, y, cfg = load(name)
    print(f"Dataset: {name} - {len(X)} patients, {X.shape[1]} features, "
          f"{y.mean():.0%} with {cfg['positive']}")
    print(f"Missing values filled in: {int(X.isna().sum().sum())}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows\n")

    # Baseline: always predicts the most common class.
    baseline = DummyClassifier(strategy="most_frequent")

    # Real model: fill missing values with the median, then a random forest.
    model = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("forest", RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE)),
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    results = {}
    for label, clf in [("Baseline (most frequent)", baseline), ("Random Forest", model)]:
        # Cross-validation on the training set only - the test set stays unseen.
        cv_scores = cross_val_score(clf, X_train, y_train, cv=cv, scoring="accuracy")
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        results[label] = (
            cv_scores.mean(), cv_scores.std(),
            accuracy_score(y_test, preds),
            recall_score(y_test, preds, zero_division=0),
        )

    print(f"{'Model':<26}{'CV accuracy':>16}{'Test accuracy':>15}{'Test recall':>13}")
    print("-" * 70)
    for label, (cv_mean, cv_std, acc, rec) in results.items():
        print(f"{label:<26}{f'{cv_mean:.1%} ± {cv_std:.1%}':>16}{acc:>15.1%}{rec:>13.1%}")

    gain = results["Random Forest"][2] - results["Baseline (most frequent)"][2]
    print(f"\nRandom Forest beats the baseline by {gain * 100:.1f} percentage points "
          "of test accuracy.")
    print(f"Recall = share of real {cfg['positive']} cases the model caught.")

    # Fairness check: does the model work equally well for each group?
    if cfg["group_col"]:
        preds = model.predict(X_test)
        print(f"\nTest results by {cfg['group_col']}:")
        for value, group in cfg["group_names"].items():
            mask = (X_test[cfg["group_col"]] == value).to_numpy()
            if mask.sum() == 0:
                continue
            acc = accuracy_score(y_test[mask], preds[mask])
            rec = recall_score(y_test[mask], preds[mask], zero_division=0)
            print(f"  {group:<8} n={mask.sum():<4} accuracy {acc:.1%}  recall {rec:.1%}")
        print("  (Small groups -> noisy numbers. Report them, but don't over-read them.)")

    # Which features mattered most?
    importances = pd.Series(
        model.named_steps["forest"].feature_importances_, index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 5 features:")
    for feat, score in importances.head(5).items():
        print(f"  {feat:<26}{score:.3f}")


if __name__ == "__main__":
    main()
