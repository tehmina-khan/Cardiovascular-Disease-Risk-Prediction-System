import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import lightgbm as lgb

from data_preprocessing import load_data, clean_data, feature_engineering, prepare_features


def train():
    df = load_data("data/cardio.csv")

    df = clean_data(df)
    df = feature_engineering(df)

    X, y = prepare_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    sm = SMOTE(random_state=42)
    X_train_res, y_train_res = sm.fit_resample(X_train_scaled, y_train)

    model = lgb.LGBMClassifier(random_state=42)
    model.fit(X_train_res, y_train_res)

    preds = model.predict_proba(X_test_scaled)[:, 1]
    print("ROC-AUC:", roc_auc_score(y_test, preds))

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "features": X.columns.tolist()
    }, "model.pkl")

    print("✅ Model saved")


if __name__ == "__main__":
    train()