import joblib


class ModelPredictor:
    def __init__(self, model_path: str):
        self.model = joblib.load(model_path)

    def predict(self, text: str):
        prediction = self.model.predict([text])[0]
        probability = self.model.predict_proba([text])[0]
        return {
            "text": text,
            "prediction": prediction,
            "probability": probability.tolist(),
        }


if __name__ == "__main__":
    predictor = ModelPredictor("model/svm_model.pkl")
    prediction = predictor.predict("wow nice and cool")
    print(prediction)
