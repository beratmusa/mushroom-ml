import os
import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# En iyi modeli yükle
MODEL_PATH = "D:/mushroom-ml/backend/model/best_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        best_model = pickle.load(file)
else:
    best_model = None

# Dinamik model sonuçları (örnek veriler train.py ile ilişkilendirilebilir)
model_results = [
  {
    "step": "Ham Veri İle Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.95,
          "F1 Score": 0.94,
          "ROC AUC": 0.96,
          "Sensitivity": 0.93,
          "Specificity": 0.96,
          "Train Loss": 0.2,
          "Test Loss": 0.25,
          "Confusion Matrix": [[50, 5], [3, 42]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.92,
          "F1 Score": 0.91,
          "ROC AUC": 0.93,
          "Sensitivity": 0.9,
          "Specificity": 0.94,
          "Train Loss": 0.3,
          "Test Loss": 0.35,
          "Confusion Matrix": [[48, 7], [4, 41]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.92,
          "F1 Score": 0.92,
          "ROC AUC": 0.92,
          "Sensitivity": 0.91,
          "Specificity": 0.93,
          "Train Loss": 0.93,
          "Test Loss": 0.92,
          "Confusion Matrix": [[49, 6], [3, 42]]
        }
      }
    ]
  },
  {
    "step": "Gürültülü Veri İle Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.93,
          "F1 Score": 0.92,
          "ROC AUC": 0.94,
          "Sensitivity": 0.91,
          "Specificity": 0.95,
          "Train Loss": 0.22,
          "Test Loss": 0.27,
          "Confusion Matrix": [[51, 4], [4, 41]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.91,
          "F1 Score": 0.9,
          "ROC AUC": 0.92,
          "Sensitivity": 0.89,
          "Specificity": 0.93,
          "Train Loss": 0.28,
          "Test Loss": 0.33,
          "Confusion Matrix": [[49, 6], [4, 41]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.91,
          "F1 Score": 0.9,
          "ROC AUC": 0.92,
          "Sensitivity": 0.89,
          "Specificity": 0.93,
          "Train Loss": 0.82,
          "Test Loss": 0.91,
          "Confusion Matrix": [[50, 5], [5, 40]]
        }
      }
    ]
  },
  {
    "step": "PCA ile Gürültü Temizleme İle Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.94,
          "F1 Score": 0.93,
          "ROC AUC": 0.95,
          "Sensitivity": 0.92,
          "Specificity": 0.96,
          "Train Loss": 0.19,
          "Test Loss": 0.23,
          "Confusion Matrix": [[52, 3], [3, 42]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.91,
          "F1 Score": 0.9,
          "ROC AUC": 0.92,
          "Sensitivity": 0.89,
          "Specificity": 0.93,
          "Train Loss": 0.28,
          "Test Loss": 0.33,
          "Confusion Matrix": [[49, 6], [4, 41]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.91,
          "F1 Score": 0.9,
          "ROC AUC": 0.92,
          "Sensitivity": 0.89,
          "Specificity": 0.93,
          "Train Loss": 0.92,
          "Test Loss": 0.91,
          "Confusion Matrix": [[50, 5], [5, 40]]
        }
      }
    ]
  },
  {
    "step": "SMOTE İle Dengesizlikle Başa Çıkma Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.96,
          "F1 Score": 0.95,
          "ROC AUC": 0.97,
          "Sensitivity": 0.94,
          "Specificity": 0.98,
          "Train Loss": 0.17,
          "Test Loss": 0.21,
          "Confusion Matrix": [[53, 2], [2, 43]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.91,
          "F1 Score": 0.9,
          "ROC AUC": 0.92,
          "Sensitivity": 0.89,
          "Specificity": 0.93,
          "Train Loss": 0.28,
          "Test Loss": 0.33,
          "Confusion Matrix": [[49, 6], [4, 41]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.94,
          "F1 Score": 0.93,
          "ROC AUC": 0.95,
          "Sensitivity": 0.92,
          "Specificity": 0.96,
          "Train Loss": 0.91,
          "Test Loss": 0.92,
          "Confusion Matrix": [[51, 4], [3, 42]]
        }
      }
    ]
  },
  {
    "step": "Normalizasyon Uygulanmış Veri İle Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.97,
          "F1 Score": 0.96,
          "ROC AUC": 0.98,
          "Sensitivity": 0.95,
          "Specificity": 0.99,
          "Train Loss": 0.15,
          "Test Loss": 0.19,
          "Confusion Matrix": [[54, 1], [1, 44]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.91,
          "F1 Score": 0.9,
          "ROC AUC": 0.92,
          "Sensitivity": 0.89,
          "Specificity": 0.93,
          "Train Loss": 0.28,
          "Test Loss": 0.33,
          "Confusion Matrix": [[49, 6], [4, 41]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.95,
          "F1 Score": 0.94,
          "ROC AUC": 0.96,
          "Sensitivity": 0.93,
          "Specificity": 0.97,
          "Train Loss": 0.93,
          "Test Loss": 0.92,
          "Confusion Matrix": [[52, 3], [2, 43]]
        }
      }
    ]
  },
  {
    "step": "K-Fold Çapraz Doğrulama İle Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.96,
          "F1 Score": 0.95,
          "ROC AUC": 0.97,
          "Sensitivity": 0.94,
          "Specificity": 0.98,
          "Train Loss": 0.16,
          "Test Loss": 0.2,
          "Confusion Matrix": [[53, 2], [2, 43]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.93,
          "F1 Score": 0.92,
          "ROC AUC": 0.94,
          "Sensitivity": 0.91,
          "Specificity": 0.95,
          "Train Loss": 0.27,
          "Test Loss": 0.32,
          "Confusion Matrix": [[50, 5], [4, 41]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.95,
          "F1 Score": 0.94,
          "ROC AUC": 0.96,
          "Sensitivity": 0.93,
          "Specificity": 0.91,
          "Train Loss": 0.27,
          "Test Loss": 0.32,
          "Confusion Matrix": [[50, 5], [4, 41]]
                }
            }
        ]
    }
]

@app.route('/results', methods=['GET'])
def get_results():
    return jsonify(model_results)

@app.route('/predict', methods=['POST'])
def predict():
    if not best_model:
        return jsonify({"error": "En iyi model yüklenemedi."}), 500

    input_data = request.json.get("input", [])
    if len(input_data) != 21:  # Özellik sayısını kontrol edin
        return jsonify({"error": "21 özellik bekleniyor."}), 400

    prediction = best_model.predict([input_data])
    return jsonify({
        "prediction": int(prediction[0]),
        "message": "Tahmin başarılı!"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
