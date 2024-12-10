import pickle

import numpy as np
from flask import Flask, jsonify, request

app = Flask(__name__)

# Yüklenen en iyi model ve scaler
model_path = "./model/best_model.pkl"
scaler_path = "./model/scaler.pkl"

with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)

with open(scaler_path, "rb") as scaler_file:
    scaler = pickle.load(scaler_file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Kullanıcıdan gelen JSON verisini al
        data = request.json
        features = np.array(data["features"]).reshape(1, -1)

        # Normalizasyon işlemi
        scaled_features = scaler.transform(features)

        # Tahmin ve olasılık değerlerini hesapla
        prediction = model.predict(scaled_features)[0]
        probabilities = model.predict_proba(scaled_features)[0]

        response = {
            "prediction": int(prediction),
            "probabilities": probabilities.tolist()
        }
        return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
