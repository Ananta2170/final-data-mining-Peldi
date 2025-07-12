import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# ------------------- SETUP ---------------------
st.set_page_config(page_title="Aplikasi Kesehatan & Lokasi", layout="wide")
st.title("🧠 Aplikasi Prediksi Diabetes & Clustering")

# Sidebar navigasi
st.sidebar.title("Navigasi Aplikasi")
halaman = st.sidebar.radio("Pilih Halaman", [
    "Klasifikasi Diabetes",
    "Clustering Pasien",
    "Clustering Gerai Kopi"
])

# ------------------- Load Dataset Diabetes ---------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes.csv")

df_diabetes = load_data()
X = df_diabetes.drop("Outcome", axis=1)
y = df_diabetes["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- KLASIFIKASI DIABETES ---------------------
if halaman == "Klasifikasi Diabetes":
    st.header("🧪 Prediksi Diabetes Menggunakan KNN")
    st.markdown("Model KNN digunakan untuk memprediksi apakah seorang pasien menderita diabetes berdasarkan fitur-fitur medis.")

    # Split & Train
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
    knn = KNeighborsClassifier(n_neighbors=7)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)

    # Evaluasi
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Metrik Evaluasi")
        report = classification_report(y_test, y_pred, output_dict=True)
        st.dataframe(pd.DataFrame(report).transpose())

    with col2:
        st.subheader("🧮 Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_xlabel("Prediksi")
        ax.set_ylabel("Aktual")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("📝 Input Data Pasien")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 1)
        glucose = st.number_input("Glucose", 0, 300, 120)
    with col2:
        blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
        skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
    with col3:
        insulin = st.number_input("Insulin", 0, 900, 80)
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    with col4:
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.number_input("Age", 0, 120, 33)

    if st.button("Prediksi"):
        input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                                insulin, bmi, dpf, age]])
        input_scaled = scaler.transform(input_data)
        prediction = knn.predict(input_scaled)
        hasil = "✅ Tidak Diabetes" if prediction[0] == 0 else "⚠️ Positif Diabetes"
        st.success(f"Hasil Prediksi: **{hasil}**")

# ------------------- CLUSTERING PASIEN ---------------------
elif halaman == "Clustering Pasien":
    st.header("📊 Clustering Pasien Menggunakan KMeans")
    st.markdown("Model KMeans digunakan untuk mengelompokkan pasien berdasarkan kemiripan fitur medis.")

    # Clustering pasien
    kmeans = KMeans(n_clusters=3, random_state=42)
    df_diabetes['Cluster'] = kmeans.fit_predict(X_scaled)

    st.subheader("📈 Visualisasi Clustering Berdasarkan Glucose dan BMI")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_diabetes, x="Glucose", y="BMI", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Clustering Pasien")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📝 Input Data Pasien Baru untuk Clustering")
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Glucose", 0, 300, 120, key="cg")
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, key="cb")
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80, key="ci")
        age = st.number_input("Age", 0, 120, 33, key="ca")

    if st.button("Clusterkan Pasien"):
        input_partial = pd.DataFrame([[glucose, bmi, insulin, age]],
                                     columns=["Glucose", "BMI", "Insulin", "Age"])
        input_full = pd.DataFrame(columns=X.columns)
        for col in X.columns:
            input_full[col] = [input_partial[col].values[0] if col in input_partial.columns else 0]

        input_scaled = scaler.transform(input_full)
        cluster_result = kmeans.predict(input_scaled)
        st.success(f"Pasien ini termasuk ke dalam **Cluster {cluster_result[0]}**")

# ------------------- CLUSTERING GERAI KOPI ---------------------
elif halaman == "Clustering Gerai Kopi":
    st.header("📍 Clustering Lokasi Gerai Kopi - KMeans")
    st.markdown("Aplikasi ini mengelompokkan lokasi gerai kopi berdasarkan koordinat X dan Y menggunakan algoritma KMeans.")

    X_kopi, _ = make_blobs(n_samples=1000, centers=3, n_features=2, random_state=42)
    df_kopi = pd.DataFrame(X_kopi, columns=["X", "Y"])

    kmeans_kopi = KMeans(n_clusters=3, random_state=42)
    df_kopi["Cluster"] = kmeans_kopi.fit_predict(df_kopi[["X", "Y"]])

    st.subheader("📈 Visualisasi Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df_kopi, x="X", y="Y", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Cluster Lokasi Gerai Kopi (Data Dummy)")
    st.pyplot(fig)

    st.subheader("📝 Input Lokasi Gerai Baru")
    x_input = st.number_input("Koordinat X", value=0.0, format="%.2f")
    y_input = st.number_input("Koordinat Y", value=0.0, format="%.2f")

    if st.button("Cek Cluster"):
        pred = kmeans_kopi.predict([[x_input, y_input]])[0]
        st.success(f"📌 Lokasi tersebut masuk ke dalam **Cluster: {pred}**")
