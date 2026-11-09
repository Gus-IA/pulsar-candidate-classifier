"""Train the final pulsar classifier and save it to disk."""

import pickle

import pandas as pd
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

DATA_PATH = "data/htru2.csv"
MODEL_PATH = "model.pkl"
RANDOM_STATE = 42
THRESHOLD = 0.58  # chosen on cross-validated predictions (see notebook)

# Best parameters found in the notebook (5-fold CV, PR-AUC)
PARAMS = {
    "n_estimators": 200,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
}


def main():
    df = pd.read_csv(DATA_PATH)
    features = [c for c in df.columns if c != "class"]

    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df["class"], random_state=RANDOM_STATE
    )
    X_train, y_train = df_train[features], df_train["class"]
    X_test, y_test = df_test[features], df_test["class"]

    # Class imbalance: weight positives by the negative/positive ratio
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    model = XGBClassifier(
        **PARAMS,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    print(f"Test PR-AUC   : {average_precision_score(y_test, proba):.4f}")
    print(f"Test precision: {precision_score(y_test, pred):.4f}")
    print(f"Test recall   : {recall_score(y_test, pred):.4f}")

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({"model": model, "threshold": THRESHOLD, "features": features}, f)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    main()
