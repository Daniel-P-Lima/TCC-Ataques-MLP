"""Quick SVM sample runner using a small subset of the pre-split data.

Use this to validate the preprocessing and training pipeline quickly.
Runs a fast `SGDClassifier` (linear hinge) on a sample of the training data.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, label_binarize
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report, roc_curve
from pathlib import Path
import joblib
import matplotlib.pyplot as plt

# Configuration
TRAIN_X = "./Data/Xtrain.csv"
TRAIN_Y = "./Data/ytrain.csv"
TEST_X = "./Data/Xtest.csv"
TEST_Y = "./Data/ytest.csv"
SAMPLE_SIZE = 5000  # number of training rows to sample (adjustable)
RANDOM_STATE = 42
NUMERIC_COLS = [
    "duration",
    "orig_bytes", "resp_bytes", "missed_bytes",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
]


def load_sample(n=SAMPLE_SIZE):
    X_train = pd.read_csv(TRAIN_X)
    y_train = pd.read_csv(TRAIN_Y)
    X_test = pd.read_csv(TEST_X)
    y_test = pd.read_csv(TEST_Y)

    # extract label columns
    if y_train.shape[1] == 1:
        y_train_raw = y_train.iloc[:, 0].values
    else:
        y_train_raw = y_train.values.ravel()
    if y_test.shape[1] == 1:
        y_test_raw = y_test.iloc[:, 0].values
    else:
        y_test_raw = y_test.values.ravel()

    # sample train set
    if n is not None and n > 0 and n < len(X_train):
        rng = np.random.default_rng(RANDOM_STATE)
        idx = rng.choice(len(X_train), size=n, replace=False)
        X_train = X_train.iloc[idx].reset_index(drop=True)
        y_train_raw = np.asarray(y_train_raw)[idx]

    # drop known redundant cols if present
    drop_cols = [c for c in ["Unnamed: 0", "id.orig_h", "id.resp_h", "id.orig_p", "id.resp_p"] if c in X_train.columns]
    if drop_cols:
        X_train = X_train.drop(columns=drop_cols)
        X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])

    # encode labels: fit on union of train and test labels to avoid unseen-label errors
    le = LabelEncoder()
    le.fit(np.concatenate([y_train_raw, y_test_raw]))
    y_train = le.transform(y_train_raw)
    y_test = le.transform(y_test_raw)

    # scale numeric cols present
    numeric_present = [c for c in NUMERIC_COLS if c in X_train.columns]
    scaler = StandardScaler()
    if numeric_present:
        X_train[numeric_present] = scaler.fit_transform(X_train[numeric_present])
        X_test[numeric_present] = scaler.transform(X_test[numeric_present])

    return X_train.values.astype(np.float32), X_test.values.astype(np.float32), y_train, y_test, scaler, le


def train_and_eval():
    print(f"Loading sample (n={SAMPLE_SIZE}) and training quick SGDClassifier...")
    X_train, X_test, y_train, y_test, scaler, le = load_sample(SAMPLE_SIZE)
    print("Data shapes:", X_train.shape, X_test.shape)

    clf = SGDClassifier(loss="hinge", max_iter=1000, tol=1e-3, class_weight="balanced", random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)

    # use decision_function for scores
    try:
        scores = clf.decision_function(X_test)
    except Exception:
        scores = clf.predict_proba(X_test)[:, 1]
    preds = clf.predict(X_test)

    if len(np.unique(y_train)) == 2:
        auc = roc_auc_score(y_test, scores)
    else:
        y_test_b = label_binarize(y_test, classes=np.unique(y_train))
        auc = roc_auc_score(y_test_b, scores, multi_class="ovr", average="macro")

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds, average="macro", zero_division=0)
    rec = recall_score(y_test, preds, average="macro", zero_division=0)

    print("-- Results --")
    print(f"accuracy: {acc:.4f}  precision: {prec:.4f}  recall: {rec:.4f}  auc: {auc:.4f}")
    print(classification_report(y_test, preds, digits=4))

    out = Path(__file__).resolve().parent
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, out / "svm_sample.joblib")
    joblib.dump(scaler, out / "svm_sample_scaler.joblib")
    joblib.dump(le, out / "svm_sample_label_encoder.joblib")
    print("Saved sample model and artifacts to", out)

    # save a small ROC plot
    plt.figure()
    if len(np.unique(y_train)) == 2:
        fpr, tpr, _ = roc_curve(y_test, scores)
        plt.plot(fpr, tpr, label=f"ROC (AUC={auc:.4f})")
    else:
        y_test_b = label_binarize(y_test, classes=np.unique(y_train))
        fpr, tpr, _ = roc_curve(y_test_b.ravel(), scores.ravel())
        plt.plot(fpr, tpr, label=f"ROC micro (AUC={auc:.4f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "svm_sample_roc.png", dpi=150)


if __name__ == "__main__":
    train_and_eval()
