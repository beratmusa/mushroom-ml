import os
import pickle

from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

# En iyi modeli yükleme
MODEL_PATH = "D:/mushroom-ml/backend/model/best_model.pkl"
if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as file:
        best_model = pickle.load(file)
else:
    best_model = None

# Dinamik model sonuçları
model_results = [
  {
    "step": "Ham Veri İle Deneysel Çalışma",
    "results": [
      {
        "model": "Logistic Regression",
        "metrics": {
          "Accuracy": 0.94,
          "F1 Score": 0.93,
          "ROC AUC": 0.97,
          "Sensitivity": 0.91,
          "Specificity": 0.96,
          "Train Loss": 0.17,
          "Test Loss": 0.18,
          "Confusion Matrix": [[869, 34], [66, 714]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.86,
          "F1 Score": 0.85,
          "ROC AUC": 0.93,
          "Sensitivity": 0.87,
          "Specificity": 0.86,
          "Train Loss": 0.73,
          "Test Loss": 0.66,
          "Confusion Matrix": [[778, 125], [99, 681]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.99,
          "F1 Score": 0.99,
          "ROC AUC": 1,
          "Sensitivity": 0.99,
          "Specificity": 1,
          "Train Loss": 0.0024,
          "Test Loss": 0.0032,
          "Confusion Matrix": [[903, 0], [3, 777]]
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
          "Accuracy": 0.92,
          "F1 Score": 0.91,
          "ROC AUC": 0.97,
          "Sensitivity": 0.89,
          "Specificity": 0.94,
          "Train Loss": 0.20,
          "Test Loss": 0.21,
          "Confusion Matrix": [[854, 49], [84, 694]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.88,
          "F1 Score": 0.87,
          "ROC AUC": 0.93,
          "Sensitivity": 0.82,
          "Specificity": 0.93,
          "Train Loss": 0.65,
          "Test Loss": 0.60,
          "Confusion Matrix": [[848, 55], [137, 643]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.99,
          "F1 Score": 0.99,
          "ROC AUC": 1,
          "Sensitivity": 0.99,
          "Specificity": 1,
          "Train Loss": 0.0036,
          "Test Loss": 0.0043,
          "Confusion Matrix": [[903, 0], [4, 776]]
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
          "Accuracy": 0.84,
          "F1 Score": 0.79,
          "ROC AUC": 0.88,
          "Sensitivity": 0.79,
          "Specificity": 0.89,
          "Train Loss": 0.37,
          "Test Loss": 0.39,
          "Confusion Matrix": [[808, 95], [161, 619]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.86,
          "F1 Score": 0.84,
          "ROC AUC": 0.88,
          "Sensitivity": 0.77,
          "Specificity": 0.94,
          "Train Loss": 0.38,
          "Test Loss": 0.40,
          "Confusion Matrix": [[853, 50], [173, 607]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.99,
          "F1 Score": 0.99,
          "ROC AUC": 0.99,
          "Sensitivity": 0.99,
          "Specificity": 0.99,
          "Train Loss": 0.0059,
          "Test Loss": 0.0091,
          "Confusion Matrix": [[899, 4], [1, 779]]
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
          "Accuracy": 0.94,
          "F1 Score": 0.93,
          "ROC AUC": 0.97,
          "Sensitivity": 0.92,
          "Specificity": 0.95,
          "Train Loss": 0.16,
          "Test Loss": 0.18,
          "Confusion Matrix": [[866, 37], [62, 718]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.86,
          "F1 Score": 0.85,
          "ROC AUC": 0.93,
          "Sensitivity": 0.87,
          "Specificity": 0.85,
          "Train Loss": 0.76,
          "Test Loss": 0.67,
          "Confusion Matrix": [[775, 128], [99, 681]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 0.99,
          "F1 Score": 0.99,
          "ROC AUC": 1,
          "Sensitivity": 0.99,
          "Specificity": 1,
          "Train Loss": 0.0022,
          "Test Loss": 0.0029,
          "Confusion Matrix": [[903, 0], [1, 779]]
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
          "Accuracy": 0.93,
          "F1 Score": 0.93,
          "ROC AUC": 0.97,
          "Sensitivity": 0.91,
          "Specificity": 0.95,
          "Train Loss": 0.18,
          "Test Loss": 0.19,
          "Confusion Matrix": [[866, 37], [69, 711]]
        }
      },
      {
        "model": "Naive Bayes",
        "metrics": {
          "Accuracy": 0.86,
          "F1 Score": 0.85,
          "ROC AUC": 0.93,
          "Sensitivity": 0.87,
          "Specificity": 0.86,
          "Train Loss": 0.73,
          "Test Loss": 0.66,
          "Confusion Matrix": [[778, 125], [99, 681]]
        }
      },
      {
        "model": "SVM",
        "metrics": {
          "Accuracy": 1,
          "F1 Score": 1,
          "ROC AUC": 1,
          "Sensitivity": 1,
          "Specificity": 1,
          "Train Loss": 0.00019,
          "Test Loss": 0.00024,
          "Confusion Matrix": [[903, 0], [0, 780]]
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
          "Accuracy": 0.94,
          "F1 Score": 0.93,
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
          "F1 Score": 0.85,
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
          "F1 Score": 0.99,
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
