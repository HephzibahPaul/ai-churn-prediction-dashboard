from pathlib import Path
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score

from .data_preprocessing import load_and_preprocess_data


def train_and_save_model():
    # Load and split data
    X_train, X_test, y_train, y_test, feature_cols = load_and_preprocess_data()

    # Define model
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    # Train
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.3f}")

    # Prepare models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    # Save model
    model_path = models_dir / "churn_model.pkl"
    feature_cols_path = models_dir / "feature_cols.pkl"
    feature_importances_path = models_dir / "feature_importances.pkl"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    with open(feature_cols_path, "wb") as f:
        pickle.dump(feature_cols, f)

    # Save feature importances as {feature_name: importance}
    importances = model.feature_importances_
    feat_imp_dict = {col: float(imp) for col, imp in zip(feature_cols, importances)}
    with open(feature_importances_path, "wb") as f:
        pickle.dump(feat_imp_dict, f)

    print(f"Model saved to {model_path.resolve()}")
    print(f"Feature columns saved to {feature_cols_path.resolve()}")
    print(f"Feature importances saved to {feature_importances_path.resolve()}")


if __name__ == "__main__":
    train_and_save_model()
