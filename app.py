from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os

app = Flask(__name__)
CORS(app)

model = joblib.load("XGBoost_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json
    return jsonify(data)
    # data = request.json["features"]
    # prediction = model.predict([data])
    # return jsonify({"prediction": float(prediction[0])})



if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)

