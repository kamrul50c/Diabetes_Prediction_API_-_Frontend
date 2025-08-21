import json, numpy as np, pandas as pd
from pathlib import Path
from joblib import dump
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

RANDOM_STATE = 42
DATA_PATH = "data/diabetes.csv"  
MODEL_PATH = "models/diabetes_model.pkl"
METRICS_PATH = "metrics.json"
METRICS_ALL_PATH = "metrics_all.json"

df = pd.read_csv(DATA_PATH)
target_col = "Outcome"
X = df.drop(columns=[target_col]).copy()
y = df[target_col].copy()

zero_as_nan = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for c in zero_as_nan:
    X.loc[X[c] == 0, c] = np.nan

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
X_fit, X_cal, y_fit, y_cal = train_test_split(
    X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

pipe_lr  = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=5000, random_state=RANDOM_STATE))])

pipe_svm = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("clf", SVC(probability=False, kernel="rbf", random_state=RANDOM_STATE))])

pipe_knn = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("scaler", StandardScaler()),
                     ("clf", KNeighborsClassifier())])

pipe_dt  = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", DecisionTreeClassifier(random_state=RANDOM_STATE))])

pipe_rf  = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("clf", RandomForestClassifier(random_state=RANDOM_STATE))])

grids = {
    "LogReg": (pipe_lr, {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__class_weight": [None, "balanced"]
    }, "grid"),
    "SVM": (pipe_svm, {
        "clf__C": np.logspace(-3, 3, 13),
        "clf__gamma": ["scale"] + list(np.logspace(-4, 1, 10))
    }, "random"),
    "KNN": (pipe_knn, {
        "clf__n_neighbors": list(range(3, 52, 2)),
        "clf__weights": ["uniform", "distance"],
        "clf__p": [1, 2]
    }, "grid"),
    "DecisionTree": (pipe_dt, {
        "clf__max_depth": [None, 3, 5, 7, 10, 15, 20],
        "clf__min_samples_leaf": [1, 2, 3, 5, 8],
        "clf__class_weight": [None, "balanced"]
    }, "grid"),
    "RandomForest": (pipe_rf, {
        "clf__n_estimators": [300, 600, 1000],
        "clf__max_depth": [None, 5, 10, 20],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", "log2", None],
        "clf__class_weight": [None, "balanced"]
    }, "random")  
}

def fit_with_search(name, pipe, params, mode):
    if mode == "random":
        search = RandomizedSearchCV(
            pipe, params, n_iter=30, cv=cv, scoring="roc_auc",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True, verbose=0
        )
    else:
        search = GridSearchCV(
            pipe, params, cv=cv, scoring="roc_auc",
            n_jobs=-1, refit=True, verbose=0
        )
    search.fit(X_fit, y_fit)
    return search.best_estimator_

def f1_threshold(y_true, p):
    thresholds = np.linspace(0.05, 0.95, 181)
    best_t, best_s = 0.5, -1
    for t in thresholds:
        y_pred = (p >= t).astype(int)
        s = f1_score(y_true, y_pred, zero_division=0)
        if s > best_s:
            best_s, best_t = s, t
    return float(best_t), float(best_s)

candidates = []
for name, (pipe, params, mode) in grids.items():
    best = fit_with_search(name, pipe, params, mode)
    calib = CalibratedClassifierCV(best, method="isotonic", cv="prefit")
    calib.fit(X_cal, y_cal)
    p_cal = calib.predict_proba(X_cal)[:, 1]
    roc_cal = roc_auc_score(y_cal, p_cal)
    thr, thr_f1 = f1_threshold(y_cal, p_cal)
    candidates.append({
        "name": name,
        "model": calib,
        "roc_auc_cal": float(roc_cal),
        "threshold": thr,
        "thr_f1_cal": thr_f1
    })

best = max(candidates, key=lambda d: d["roc_auc_cal"])

metrics_all = []
for c in candidates:
    model = c["model"]
    t = c["threshold"]
    p_test = model.predict_proba(X_test)[:, 1]
    y_pred = (p_test >= t).astype(int)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc = roc_auc_score(y_test, p_test)
    metrics_all.append({
        "model": c["name"],
        "threshold": round(t, 4),
        "cal_roc_auc": round(c["roc_auc_cal"], 4),
        "test_accuracy": round(acc, 4),
        "test_precision": round(prec, 4),
        "test_recall": round(rec, 4),
        "test_f1": round(f1, 4),
        "test_roc_auc": round(roc, 4)
    })

best_model = next(m for m in candidates if m["name"] == best["name"])
p_test_best = best_model["model"].predict_proba(X_test)[:, 1]
y_pred_best = (p_test_best >= best_model["threshold"]).astype(int)
metrics_best = {
    "model": best_model["name"],
    "threshold": round(best_model["threshold"], 4),
    "cal_roc_auc": round(best_model["roc_auc_cal"], 4),
    "test_accuracy": round(accuracy_score(y_test, y_pred_best), 4),
    "test_precision": round(precision_score(y_test, y_pred_best, zero_division=0), 4),
    "test_recall": round(recall_score(y_test, y_pred_best, zero_division=0), 4),
    "test_f1": round(f1_score(y_test, y_pred_best, zero_division=0), 4),
    "test_roc_auc": round(roc_auc_score(y_test, p_test_best), 4)
}

artifact = {
    "model": best_model["model"],           
    "threshold": float(best_model["threshold"]),
    "feature_order": list(X.columns),
    "metrics": metrics_best
}
dump(artifact, MODEL_PATH)

with open(METRICS_PATH, "w") as f:
    json.dump(metrics_best, f, indent=2)
with open(METRICS_ALL_PATH, "w") as f:
    json.dump(sorted(metrics_all, key=lambda d: d["test_roc_auc"], reverse=True), f, indent=2)

print("Best model:", json.dumps(metrics_best, indent=2))
print(f"Saved {MODEL_PATH}, {METRICS_PATH}, {METRICS_ALL_PATH}")
