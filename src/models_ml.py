"""The seven classical ML models compared in the paper, with fixed hyper-parameters.

Hyper-parameters are library defaults except where noted; they are not tuned on
the test folds. Scale-sensitive models (LR, SVM, KNN) get a StandardScaler.
"""
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from common import SEED


def score_and_predict(model, X):
    """Probability of phishing (or SVM decision value) and the 0/1 decision."""
    last = model.steps[-1][1] if hasattr(model, "steps") else model
    if isinstance(last, SVC) and not last.probability:
        s = model.decision_function(X)
        return s, (s >= 0).astype(int)
    s = model.predict_proba(X)[:, 1]
    return s, (s >= 0.5).astype(int)


def ml_models(n_jobs: int = 1, svm_probability: bool = True) -> dict:
    return {
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=SEED)),
        "Decision Tree": DecisionTreeClassifier(random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, n_jobs=n_jobs, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=200, random_state=SEED),
        "SVM (RBF)": make_pipeline(
            StandardScaler(), SVC(kernel="rbf", C=1.0, gamma="scale",
                                  probability=svm_probability, random_state=SEED)),
        "KNN (k=5)": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1, subsample=1.0,
            tree_method="hist", n_jobs=n_jobs, random_state=SEED, eval_metric="logloss"),
    }
