from preprocessing import preprocess_data
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import yaml
import os


current_dir = os.path.dirname(os.path.abspath(__file__))
params_path = os.path.join(current_dir, "params.yaml")
data_path = os.path.join(current_dir,"walmart.csv")

# Open params.yaml using the correct path
with open(params_path, "r") as f:
    params = yaml.safe_load(f)

os.makedirs(os.path.dirname("/tmp/scaler.pkl"), exist_ok=True)
os.makedirs(os.path.dirname("/tmp/random_forest_model.pkl"), exist_ok=True)


def train_model():

    # Load dataset
    #new comment
    #new comment added to trigger pipeline
    data = pd.read_csv(data_path)

    # Preprocess data
    X,y = preprocess_data(data, is_training=True, scaler_path="/tmp/scaler.pkl")

    # Train the model
    model = RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        random_state=params["random_state"]
    )
    model.fit(X, y)

    # Save the model
    model_path = "/tmp/random_forest_model.pkl"
    joblib.dump(model, model_path)
    # Comment to trigger my pipeline

if __name__ == "__main__":
    train_model()