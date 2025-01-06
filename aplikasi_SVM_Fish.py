import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler

# Load saved model and scaler
model_file = 'model_svm_fish.sav'

try:
    with open(model_file, 'rb') as f:
        svm_model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file tidak ditemukan. Pastikan 'model_svm_fish.sav' ada di direktori.")
    st.stop()

# App title
st.title("Fish Species Prediction App")
st.write("Aplikasi ini menggunakan model SVM untuk memprediksi jenis ikan berdasarkan fitur panjang, berat, dan rasio panjang-lebar.")

# Sidebar for user input
st.sidebar.header("Input Data Ikan")
length = st.sidebar.number_input("Masukkan panjang ikan (length):", min_value=0.0, step=0.1)
weight = st.sidebar.number_input("Masukkan berat ikan (weight):", min_value=0.0, step=0.1)
w_l_ratio = st.sidebar.number_input("Masukkan rasio panjang-lebar ikan (w/l ratio):", min_value=0.0, step=0.01)

# Load dataset for visualization and species mapping
def load_data():
    df_fish = pd.read_csv('fish_data.csv')
    return df_fish

try:
    df_fish = load_data()
    # Create a mapping from numerical index to species name
    species_mapping = {i: species for i, species in enumerate(df_fish['species'].unique())}
except FileNotFoundError:
    st.error("Dataset tidak ditemukan. Pastikan 'fish_data.csv' ada di direktori.")
    st.stop()

# Display dataset
if st.checkbox("Tampilkan dataset:"):
    st.write(df_fish.head())

# Display dataset description
if st.checkbox("Tampilkan deskripsi data:"):
    st.write(df_fish.describe())

# Visualize species distribution
if st.checkbox("Tampilkan distribusi species (pie chart):"):
    st.subheader("Distribusi Species Ikan")
    species_count = df_fish['species'].value_counts()
    fig, ax = plt.subplots()
    species_count.plot(kind='pie', autopct='%1.1f%%', ax=ax)
    plt.title("Persentase Species Ikan")
    st.pyplot(fig)

# Correlation heatmap
if st.checkbox("Tampilkan heatmap korelasi:"):
    st.subheader("Heatmap Korelasi")
    numeric_df = df_fish.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(numeric_df.corr(), cmap='BuPu', annot=True, ax=ax)
    st.pyplot(fig)

# Prediction section
st.header("Prediksi Species Ikan")
if st.button("Prediksi"):
    # Prepare input data
    input_data = pd.DataFrame({'length': [length], 'weight': [weight], 'w_l_ratio': [w_l_ratio]})
    try:
        # Standardize using the same scaler (previously fitted on training data)
        scaler = StandardScaler()
        #Fit scaler only once on the loaded dataset. Avoid fitting the scaler again on the input data
        scaled_data = scaler.fit_transform(df_fish.drop(columns=['species']))
        scaled_input = scaler.transform(input_data)

        # Perform prediction
        prediction = svm_model.predict(scaled_input)[0] #Get the first element from the prediction array
        predicted_species = species_mapping[prediction] #Map the prediction to the species name
        st.success(f"Jenis ikan yang diprediksi: {predicted_species}")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat melakukan prediksi: {e}")