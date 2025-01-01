import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             confusion_matrix, f1_score, log_loss,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils import shuffle

# Veriyi yükleme
data = pd.read_csv("D:/mushroom-dm/data/mushroom_data.csv")

# PARTIAL sütununu çıkarma
if 'PARTIAL' in data.columns:
    data = data.drop(columns=['PARTIAL'])
    print("PARTIAL sütunu veri setinden çıkarıldı.")

target_column = data.columns[0]  # İlk sütun hedef sınıf
X = data.iloc[:, 1:]  # Özellik sütunları
y = data[target_column]  # Hedef sınıf

# Kategorik veriyi sayısallaştırma
X = X.apply(LabelEncoder().fit_transform)
y = LabelEncoder().fit_transform(y)

# MinMaxScaler ile standardizasyon
scaler = MinMaxScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=data.columns[1:])

def remove_correlated_features(data, high_threshold=0.8, low_threshold=0.1):
    # Korelasyon matrisi oluşturuluyor
    correlation_matrix = data.corr().abs()
    
    # Üst üçgen matrisi alınarak tekrarlamalar engelleniyor
    upper_triangle = correlation_matrix.where(
        np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool)
    )
    
    # Yüksek korelasyonlu sütunları belirleme
    high_corr_to_drop = [
        column for column in upper_triangle.columns 
        if any(upper_triangle[column] > high_threshold)
    ]
    
    # Düşük korelasyonlu sütunları belirleme
    low_corr_to_drop = [
        column for column in correlation_matrix.columns 
        if all(correlation_matrix[column] < low_threshold)
    ]
    
    # Hem yüksek hem düşük korelasyonlu sütunları kaldırma
    to_drop = set(high_corr_to_drop + low_corr_to_drop)
    data = data.drop(columns=to_drop)
    
    print(f"Yüksek korelasyon nedeniyle çıkarılan sütunlar: {high_corr_to_drop}")
    print(f"Düşük korelasyon nedeniyle çıkarılan sütunlar: {low_corr_to_drop}")
    
    return data

# Korelasyon analizi sonrası yüksek korelasyonlu sütunları çıkarma
X = remove_correlated_features(X)

# Korelasyon Analizi
def plot_correlation_matrix(data):
    correlation_matrix = data.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Korelasyon Matrisi")
    plt.show()

# Korelasyon Matrisi başlangıçta bir kez
plot_correlation_matrix(pd.DataFrame(X, columns=X.columns))

# Overfitting Analizi (Eğitim ve Test Loss Grafiği)
def plot_loss_curves(train_loss, test_loss):
    plt.plot(train_loss, label="Eğitim Loss")
    plt.plot(test_loss, label="Test Loss")
    plt.title("Loss Eğrisi")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

# Karışıklık Matrisi
def plot_confusion_matrix(y_test, y_pred, labels):
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=labels, cmap="Blues"
    )
    plt.title("Karışıklık Matrisi")
    plt.show()

# ROC Eğrisi
def plot_roc_curve(y_test, y_pred_proba):
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="blue", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
    plt.plot([0, 1], [0, 1], color="gray", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Eğrisi")
    plt.legend(loc="lower right")
    plt.show()

# Model değerlendirme fonksiyonu
def evaluate_model(model, X_train, X_test, y_train, y_test, labels):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)

    # Loss hesaplama
    if hasattr(model, "predict_proba"):
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        test_loss = log_loss(y_test, model.predict_proba(X_test))
    else:
        train_loss, test_loss = None, None

    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)
    specificity = recall_score(y_test, y_pred, pos_label=0)
    f1 = f1_score(y_test, y_pred)

    # ROC AUC hesaplama
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None

    # Grafikler
    plot_confusion_matrix(y_test, y_pred, labels)
    if hasattr(model, "predict_proba"):
        plot_roc_curve(y_test, y_pred_proba)
    if train_loss is not None and test_loss is not None:
        plot_loss_curves([train_loss], [test_loss])

    return {
        "Karışıklık Matrisi": cm,
        "Doğruluk": accuracy,
        "Duyarlılık": sensitivity,
        "Özgüllük": specificity,
        "F1 Skoru": f1,
        "Eğitim Loss": train_loss,
        "Test Loss": test_loss,
        "ROC AUC": roc_auc,
    }

# Eğitim ve test verisini ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SMOTE ile veri dengesi sağlama
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# PCA ile boyut azaltma
pca = PCA(n_components=10, random_state=42)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)

# Modelleri tanımlama
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB()
}

# En iyi modeli seçmek için başlangıç değerleri
best_model_name = None
best_f1_score = -1
labels = ["Edible", "Poisonous"]

for name, model in models.items():
    print(f"\n{name} Sonuçları:")
    results = evaluate_model(model, X_train, X_test, y_train, y_test, labels)
    for metric, value in results.items():
        print(f"{metric}: {value}")
    if results["F1 Skoru"] > best_f1_score:
        best_f1_score = results["F1 Skoru"]
        best_model_name = name

print(f"\nEn İyi Model: {best_model_name} (F1 Skoru: {best_f1_score})")

# En iyi modeli kaydetme
best_model = models[best_model_name]
file_path = "D:/mushroom-dm/backend/model/best_model_v2.pkl"
with open(file_path, "wb") as file:
    pickle.dump(best_model, file)

print(f"En iyi model '{file_path}' olarak kaydedildi.")
