"""Main script"""

import argparse

import mlflow
import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor


def load_data(file_path: str):
    """Load data from file"""

    df = pd.read_csv(file_path)

    y = df["quality"]
    X = df.drop(columns=["quality"])

    return train_test_split(X, y, test_size=0.2, random_state=42)


def main():
    """Main function"""

    # -------------------------------------------------------------------------
    # Configuración de MLflow - DEBE ir ANTES del entrenamiento
    mlflow.set_tracking_uri("sqlite:///mlruns.db")  # Usa ruta relativa
    mlflow.set_experiment("Homework")
    # -------------------------------------------------------------------------

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--n_neighbors", type=int, default=5)
    args = parser.parse_args()

    X_train, X_test, y_train, y_test = load_data("data/winequality-red.csv")

    # -------------------------------------------------------------------------
    mlflow.set_tracking_uri("file:mlruns")
    mlflow.set_experiment("wine-quality-prediction")
    run_name = f"{args.model}_model_run"
    with mlflow.start_run(run_name=run_name):

        if args.model == "knn":
            model = KNeighborsRegressor(n_neighbors=args.n_neighbors)
            mlflow.log_param("n_neighbors", args.n_neighbors)
            mlflow.log_param("model", "knn")
        elif args.model == "elasticnet":
            model = ElasticNet()
            mlflow.log_param("model", "elasticnet")
        else:
            raise ValueError(f"Model {args.model} not supported.")

        model.fit(X_train, y_train)

        y_pred = model.predict(X_train)
        mse = mean_squared_error(y_train, y_pred)
        mae = mean_absolute_error(y_train, y_pred)
        r2 = r2_score(y_train, y_pred)
        mlflow.log_metric("train_mse", mse)
        mlflow.log_metric("train_mae", mae)
        mlflow.log_metric("train_r2", r2)
        print("Training metrics:")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        mlflow.log_metric("test_mse", mse)
        mlflow.log_metric("test_mae", mae)
        mlflow.log_metric("test_r2", r2)
        print("Testing metrics:")
        print(f"  MSE: {mse}")
        print(f"  MAE: {mae}")
        print(f"  R2: {r2}")

        # Registra el modelo usando 'registered_model_name' en lugar de 'artifact_path'
        mlflow.sklearn.log_model(
            model, artifact_path="model", registered_model_name=args.model
        )
    # -------------------------------------------------------------------------


if __name__ == "__main__":
    main()
