import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, auc, confusion_matrix, f1_score,
                             log_loss, recall_score, roc_auc_score, roc_curve)
from sklearn.model_selection import (StratifiedKFold, cross_val_score,
                                     train_test_split)
from sklearn.naive_bayes import GaussianNB  # Naive Bayes modeli
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import \
    DecisionTreeClassifier  # Decision Tree modelini import ediyoruz
from sklearn.utils import shuffle

# Veriyi yükle
data = pd.read_csv("D:/mushroom-ml/data/mushroom_data.csv")  # Veri dosyasını kontrol et

# PARTIAL sütununu çıkar
if 'PARTIAL' in data.columns:
    data = data.drop(columns=['PARTIAL'])
    print("PARTIAL sütunu veri setinden çıkarıldı.")

target_column = data.columns[0]  # İlk sütun hedef sınıf
X = data.iloc[:, 1:]  # Özellik sütunları
y = data[target_column]  # Hedef sınıf

# Kategorik veriyi sayısallaştır (Label Encoding)
X = X.apply(LabelEncoder().fit_transform)
y = LabelEncoder().fit_transform(y)

# Model değerlendirme fonksiyonu (Loss eklendi)
def evaluate_model(model, X_train, X_test, y_train, y_test):
    # Modeli eğitim verisi ile eğit
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_train_pred = model.predict(X_train)
    
    # Loss hesaplama
    if hasattr(model, "predict_proba"):
        train_loss = log_loss(y_train, model.predict_proba(X_train))
        test_loss = log_loss(y_test, model.predict_proba(X_test))
    else:
        train_loss, test_loss = None, None  # Loss hesaplanamıyorsa None döner
    
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    sensitivity = recall_score(y_test, y_pred, pos_label=1)  # Duyarlılık
    specificity = recall_score(y_test, y_pred, pos_label=0)  # Özgüllük
    f1 = f1_score(y_test, y_pred)
    
    # ROC AUC hesaplama
    if hasattr(model, "predict_proba"):
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Pozitif sınıfın olasılıkları
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    else:
        roc_auc = None
    
    return {
        "Karışıklık Matrisi": cm,
        "Doğruluk": accuracy,
        "Duyarlılık": sensitivity,
        "Özgüllük": specificity,
        "F1 Skoru": f1,
        "Eğitim Loss": train_loss,
        "Test Loss": test_loss,
        "ROC AUC": roc_auc
    }

# Model sonuçlarını yazdırmak için ortak fonksiyon (Loss sonuçlarını da içerir)
def train_and_evaluate(models, X_train, X_test, y_train, y_test, preprocessor=None):
    if preprocessor:
        X_train, X_test = preprocessor(X_train, X_test)
    
    for name, model in models.items():
        results = evaluate_model(model, X_train, X_test, y_train, y_test)
        print(f"\n{name} Sonuçları:")
        for metric, value in results.items():
            print(f"{metric}: {value}")
        
        # Overfitting kontrolü
        if results["Eğitim Loss"] is not None and results["Test Loss"] is not None:
            if results["Test Loss"] > results["Eğitim Loss"] * 1.2:  # %20 veya daha fazla fark
                print(f"Uyarı: {name} modeli overfitting yapıyor olabilir!")

# Eğitim ve test verisini ayır
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Pandas DataFrame olarak eğitim ve test verilerini yeniden yapılandıralım
X_train = pd.DataFrame(X_train, columns=X.columns)
X_test = pd.DataFrame(X_test, columns=X.columns)

# Modelleri tanımlama (Naive Bayes ekliyoruz)
models = {
    "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
    # "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "SVM": SVC(probability=True, random_state=42),
    "Naive Bayes": GaussianNB()  # Naive Bayes modeli eklendi
}

# 1. Ham veri ile model eğitimi
print("1. Ham Veri ile Model Sonuçları:")
train_and_evaluate(models, X_train, X_test, y_train, y_test)

# 2. Gürültü ekleme
rng = np.random.RandomState(42)
X_train_noisy = X_train + rng.normal(0, 0.1, X_train.shape)  # Eğitim setine gürültü ekle
X_test_noisy = X_test + rng.normal(0, 0.1, X_test.shape)  # Test setine gürültü ekle

# Gürültülü veri ile model performansı
print("\n2. Gürültülü Veri ile Model Sonuçları:")
train_and_evaluate(models, X_train_noisy, X_test_noisy, y_train, y_test)

# Gürültüyü temizleme (PCA ile boyut indirgeme)
print("\n3. Gürültü Temizlenmiş Veri ile Model Sonuçları (PCA Uygulandı):")
pca = PCA(n_components=0.95)  # Verinin %95'ini açıklayan bileşenler seçilecek

# PCA uygulayarak gürültüden temizlenmiş veri oluştur
X_train_cleaned = pca.fit_transform(X_train_noisy)
X_test_cleaned = pca.transform(X_test_noisy)

# Gürültü temizlenmiş veri ile model performansı
train_and_evaluate(models, X_train_cleaned, X_test_cleaned, y_train, y_test)

# 4. Dengesizlikle Baş Etme (SMOTE)
print("\n4. Dengesizlikle Baş Etme (SMOTE) ile Model Sonuçları:")
smote = SMOTE(sampling_strategy='auto', k_neighbors=1)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

train_and_evaluate(models, X_train_smote, X_test, y_train_smote, y_test)

# 5. Normalizasyon (Min-Max Scaling)
print("\n5. Normalizasyon (Min-Max Scaling) ile Model Sonuçları:")
scaler = MinMaxScaler()
train_and_evaluate(models, X_train, X_test, y_train, y_test, preprocessor=lambda X_train, X_test: (scaler.fit_transform(X_train), scaler.transform(X_test)))

# 6. K-Fold Çapraz Doğrulama
print("\n6. K-Fold Çapraz Doğrulama ile Model Sonuçları:")
kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for name, model in models.items():
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    print(f"\n{name} - K-Fold Çapraz Doğrulama Sonuçları: {cv_results.mean()}")

# En iyi modelin seçimi (F1 Skoru'na göre)
best_model_name = None
best_f1_score = -1  # Başlangıçta negatif bir değer

for name, model in models.items():
    results = evaluate_model(model, X_train, X_test, y_train, y_test)
    f1 = results["F1 Skoru"]
    print(f"{name} için F1 Skoru: {f1}")
    if f1 > best_f1_score:
        best_f1_score = f1
        best_model_name = name

print(f"\nEn İyi Model: {best_model_name} (F1 Skoru: {best_f1_score})")

# En iyi modeli kaydetme
best_model = models[best_model_name]
file_path = "D:/mushroom-ml/backend/model/best_model.pkl"

# # Dosya var mı kontrol et
if os.path.exists(file_path):
    print(f"{file_path} zaten mevcut, dosya adı değiştirilerek kaydedilecek.")
    file_path = "D:/mushroom-ml/backend/model/best_model_v2.pkl"  # Yeni dosya ismi

# # En iyi modeli kaydetme
with open(file_path, "wb") as file:
    pickle.dump(best_model, file)

print(f"\nEn iyi model '{file_path}' olarak kaydedildi.")
# En iyi modeli kaydet
