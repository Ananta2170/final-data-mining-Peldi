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

# ------------------- SETUP ---------------------
st.set_page_config(page_title="Diabetes App", layout="wide")
df = pd.read_csv("diabetes.csv")

X = df.drop("Outcome", axis=1)
y = df["Outcome"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Model KNN
knn = KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)

# Model KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
df['Cluster'] = kmeans.labels_

# ------------------- NAVIGASI ---------------------
st.sidebar.title("Navigasi Aplikasi")
menu = st.sidebar.radio("Pilih Menu", ["Klasifikasi Diabetes", "Clustering Pasien"])

# ------------------- KLASIFIKASI ---------------------
if menu == "Klasifikasi Diabetes":
    st.title("🧪 Prediksi Diabetes Menggunakan KNN")
    st.markdown("Model KNN digunakan untuk memprediksi apakah seorang pasien menderita diabetes berdasarkan fitur-fitur medis.")

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

# ------------------- CLUSTERING ---------------------
elif menu == "Clustering Pasien":
    st.title("📊 Clustering Pasien Menggunakan KMeans")
    st.markdown("Model KMeans digunakan untuk mengelompokkan pasien berdasarkan kemiripan fitur medis.")

    st.subheader("📈 Visualisasi Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x="Glucose", y="BMI", hue="Cluster", palette="Set2", ax=ax)
    ax.set_title("Clustering Berdasarkan Glucose dan BMI")
    st.pyplot(fig)

    st.markdown("---")
    st.subheader("📝 Input Data Baru untuk Clustering")
    col1, col2 = st.columns(2)
    with col1:
        glucose = st.number_input("Glucose", 0, 300, 120, key="cg")
        bmi = st.number_input("BMI", 0.0, 70.0, 25.0, key="cb")
    with col2:
        insulin = st.number_input("Insulin", 0, 900, 80, key="ci")
        age = st.number_input("Age", 0, 120, 33, key="ca")

    if st.button("Clusterkan"):
        cluster_input = pd.DataFrame([[glucose, bmi, insulin, age]],
                                     columns=["Glucose", "BMI", "Insulin", "Age"])
        new_input_full = pd.DataFrame(columns=X.columns)
        for col in X.columns:
            new_input_full[col] = [cluster_input[col].values[0] if col in cluster_input.columns else 0]

        new_input_scaled = scaler.transform(new_input_full)
        cluster_result = kmeans.predict(new_input_scaled)
        st.success(f"Pasien ini termasuk ke dalam **Cluster {cluster_result[0]}**")
