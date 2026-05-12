import joblib
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from models.KNN import KNNModel
from models.SVM import SVMModel
from models.RF import RFModel
from models.LiR import LogModel
from models.LiR import LinModel
from models.XGB import XGBModel
from models.LGBM import LGBMModel
from utils.data_loader import get_data
from evaluation.evaluate import evaluate
from evaluation.plotter import plot_class_distribution


FEATURE_SOURCE = "wav2vec"  # "librosa" hoặc "wav2vec"

MODEL_DIR = config.BASE_DIR / "models" / "saved"
MODEL_DIR.mkdir(parents = True, exist_ok = True)


def save_model(model, name, source):
    path = MODEL_DIR / f"{name}_{source}.joblib"
    joblib.dump(model, path)
    print(f"\nSaved → {path}")


def train_lin(X_train, y_train, X_val, y_val, X_test, y_test, source="librosa"):
    linreg = LinModel()

    linreg.find_best_params(X_train, y_train, X_val, y_val)
    y_pred = linreg.predict(X_test)
    
    evaluate(y_test, y_pred, f"Linear Regression (fit_intercept={linreg.fit_intercept})", source)
    save_model(linreg, "linreg", source)
    return linreg


def train_log(X_train, y_train, X_val, y_val, X_test, y_test, source="librosa"):
    logis = LogModel()
    logis.find_best_params(X_train, y_train, X_val, y_val)

    y_pred = logis.predict(X_test)
    evaluate(y_test, y_pred, f"Logistic Regression (C={logis.C}, max_iter={logis.max_iter})", source)
    save_model(logis, "log", source)


def train_knn(X_train, y_train, X_val, y_val, X_test, y_test, source = "librosa"):
    print(f"\n Tìm k tốt nhất cho KNN...")

    knn = KNNModel()
    knn.find_best_k(X_train, y_train, X_val, y_val)

    y_pred = knn.predict(X_test)
    evaluate(y_test, y_pred, f"KNN (k={knn.n_neighbors})", source)
    save_model(knn, "knn", source)

    return knn


def train_svm(X_train, y_train, X_val, y_val, X_test, y_test, source = "librosa"):
    print(f"\n Grid search cho SVM...")

    svm = SVMModel()
    svm.find_best_params(X_train, y_train, X_val, y_val)

    y_pred = svm.predict(X_test)
    evaluate(y_test, y_pred, f"SVM (C={svm.C}, gamma={svm.gamma})", source)
    save_model(svm, "svm", source)

    return svm


def train_rf(X_train, y_train, X_val, y_val, X_test, y_test, source = "librosa"):
    print(f"\n Grid search cho Random Forest...")

    rf = RFModel()
    rf.find_best_params(X_train, y_train, X_val, y_val)

    y_pred = rf.predict(X_test)
    evaluate(y_test, y_pred, f"RF (n={rf.n_estimators}, depth={rf.max_depth})", source)
    save_model(rf, "rf", source)

    return rf


def train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, source="librosa"):
    xgb = XGBModel()
    xgb.find_best_params(X_train, y_train, X_val, y_val)

    y_pred = xgb.predict(X_test)
    evaluate(y_test, y_pred, f"XGBoost (max_depth={xgb.max_depth}, lr={xgb.learning_rate})", source)
    save_model(xgb, "xgb", source)

    return xgb


def train_lgbm(X_train, y_train, X_val, y_val, X_test, y_test, source="librosa"):
    lgbm = LGBMModel()
    lgbm.find_best_params(X_train, y_train, X_val, y_val)

    y_pred = lgbm.predict(X_test)
    evaluate(y_test, y_pred, f"LightGBM (max_depth={lgbm.max_depth}, lr={lgbm.learning_rate})", source)
    save_model(lgbm, "lgbm", source)

    return lgbm


if __name__ == "__main__":
    X_train, y_train, X_val, y_val, X_test, y_test = get_data(source = FEATURE_SOURCE)

    plot_class_distribution(y_train, y_test, FEATURE_SOURCE)

    # train_lin(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
    # train_log(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
    # train_knn(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
    train_svm(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
    # train_rf(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
    # train_xgb(X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
    # train_lgbm    (X_train, y_train, X_val, y_val, X_test, y_test, source = FEATURE_SOURCE)
