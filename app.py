from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

model = joblib.load("XGBoost_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    prediction = model.predict([data])
    return jsonify({"prediction": float(prediction[0])})

if __name__ == "__main__":
    app.run()
