import mlflow
import pandas as pd

# Configura el tracking URI para que coincida con el de entrenamiento
mlflow.set_tracking_uri("sqlite:///mlruns.db")

FILE_PATH = "data/winequality-red.csv"

df = pd.read_csv(FILE_PATH)

y = df["quality"]
X = df.drop(columns=["quality"])

# Carga el modelo usando el run_id y el artifact_path
logged_model = "runs:/23130ed670f648a0990d32f5b0c77507/model"
loaded_model = mlflow.pyfunc.load_model(logged_model)
predictions = loaded_model.predict(X)
print(predictions)
