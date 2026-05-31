from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
app = Flask(__name__)
CORS(app)
model = joblib.load("XGBoost_model.pkl")
@app.route("/")
def home():
   return "API Running"
@app.route("/predict", methods=["POST"])
def predict():
   try:
       user = request.json
       # Model expects 39 features
       features = [0] * 39
       # Fill only important features
       features[18] = float(user["burnout_risk"])
       features[1] = float(user["social_media_hours"])
       features[3] = float(user["app_switch_frequency"])
       features[30] = float(user["distraction_burden_work"])
       features[7] = float(user["deep_work_hours"])
       features[6] = float(user["focus_sessions"])
       features[9] = float(user["concentration_score"])
       features[8] = float(user["task_completion_rate"])
       features[14] = float(user["motivation_level"])
       prediction = model.predict([features])
       return jsonify({
           "prediction": float(prediction[0])
       })
   except Exception as e:
       return jsonify({
           "error": str(e)
       }), 500

if __name__ == "__main__":
   port = int(os.environ.get("PORT", 5000))
   app.run(
       host="0.0.0.0",
       port=port
   )