import joblib
import pandas as pd


class Predictor:
    def __init__(self):
        data = joblib.load("model.pkl")
        self.model = data["model"]
        self.scaler = data["scaler"]
        self.features = data["features"]

    def predict(self, input_dict):
        df = pd.DataFrame([input_dict])

        # Feature engineering
        df['BMI'] = df['weight'] / ((df['height'] / 100) ** 2)
        df['pulse_pressure'] = df['ap_hi'] - df['ap_lo']

        df = df[self.features]

        df_scaled = self.scaler.transform(df)

        prob = self.model.predict_proba(df_scaled)[0][1]
        return prob


# Test
if __name__ == "__main__":
    predictor = Predictor()

    sample = {
        "age": 50,
        "gender": 1,
        "height": 170,
        "weight": 70,
        "ap_hi": 120,
        "ap_lo": 80,
        "cholesterol": 1,
        "gluc": 1,
        "smoke": 0,
        "alco": 0,
        "active": 1
    }

    print("Risk:", predictor.predict(sample))