import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "delivery_time_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_delivery_time(features):
    return model.predict([features])[0]