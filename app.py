import streamlit as st
import joblib
import pandas as pd

# Load the pkl files

preprocessor = joblib.load('preprocessor.pkl')
best_model = joblib.load('best_model.pkl')

df = pd.read_csv('data.csv')

X = df.drop(columns = ['spec_rating', 'price'])

# Unique value for each feature

unique_values = {}
for col in X.columns:
    unique_values[col] = X[col].unique().tolist()

# Let's define the input and output

def predict_laptop_price(brand, name, cpu, processor, ram, ram_type, storage, storage_type, gpu, display_size, resolution_width, resolution_height, os, warranty):
    input_data = pd.DataFrame({
        'brand': [brand],
        'name': [name],
        'processor': [processor],
        'CPU': [cpu],
        'Ram': [ram],
        'Ram_type': [ram_type],
        'ROM': [storage],
        'ROM_type': [storage_type],
        'GPU': [gpu],
        'display_size': [display_size],
        'resolution_width': [resolution_width],
        'resolution_height': [resolution_height],
        'OS': [os],
        'warranty': [warranty]
    })

    input_data_transformed = preprocessor.transform(input_data)
    price_prediction = best_model.predict(input_data_transformed)[0]

    return price_prediction


st.set_page_config(page_title="Laptop Price Predictor", page_icon="💻")
st.title("Laptop Price Predictor")
st.subheader("Enter laptop details to get predicted price")

# Define the input fields
with st.sidebar:
    st.header("Input Features")
    brand = st.selectbox("Select Brand", unique_values['brand'])
    name = st.selectbox("Enter Laptop Name", unique_values['name'])
    processor = st.selectbox("Select Processor", unique_values['processor'])
    cpu = st.selectbox("Select CPU", unique_values['CPU'])
    ram = st.selectbox("Select RAM", unique_values['Ram'])
    ram_type = st.selectbox("Select RAM Type", unique_values['Ram_type'])
    storage = st.selectbox("Select Storage", unique_values['ROM'])
    storage_type = st.selectbox("Select Storage Type", unique_values['ROM_type'])
    gpu = st.selectbox("Select GPU", unique_values['GPU'])
    display_size = st.selectbox("Select Display Size", unique_values['display_size'])
    resolution_width = st.number_input("Enter Resolution Width", min_value=float(min(unique_values['resolution_width'])), step=1.0)
    resolution_height = st.number_input("Enter Resolution Height", min_value=float(min(unique_values['resolution_height'])), step=1.0)
    os = st.selectbox("Select Operating System", unique_values['OS'])
    warranty = st.selectbox("Select Warranty", unique_values['warranty'])

# Predict the laptop price
if st.button("Predict Laptop Price"):
    results = predict_laptop_price(
        brand, name, cpu, processor, ram, ram_type, storage, storage_type, gpu, display_size, resolution_width, resolution_height, os, warranty
    )
    st.write(f"Predicted Laptop Price: ₹{results:.2f}")
    st.write(f"Predicted Laptop Price: ${results*0.012:.2f}")