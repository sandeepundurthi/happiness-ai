import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

INPUT_PATH = Path("data/processed/happiness_clean.csv")
MODEL_PATH = Path("models/happiness_xgb.pkl")


def train_model():
    df = pd.read_csv(INPUT_PATH)

    features = [
        "explained_log_gdp_per_capita",
        "explained_social_support",
        "explained_healthy_life_expectancy",
        "explained_freedom",
        "explained_generosity",
        "explained_corruption",
        "dystopia_plus_residual"
    ]

    X = df[features]
    y = df["happiness_score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = XGBRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, preds)

    mse = mean_squared_error(y_test, preds)
    rmse = mse ** 0.5

    r2 = r2_score(y_test, preds)

    print("Model Performance")
    print("-" * 40)
    print(f"MAE  : {mae:.3f}")
    print(f"RMSE : {rmse:.3f}")
    print(f"R2   : {r2:.3f}")

    importance = pd.DataFrame({
        "feature": features,
        "importance": model.feature_importances_
    }).sort_values("importance", ascending=False)

    print("\nFeature Importance:")
    print(importance)

    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()
