import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
import joblib

# Load trained model and scaler
model = load_model("iris_ann.h5")      # make sure iris_ann.h5 is in the same folder
scaler = joblib.load("scaler.pkl")     # same StandardScaler used during training

# Map encoded labels back to species names
species_labels = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# Streamlit UI
st.title("🌸 Iris Flower Classifier (ANN)")
st.write("Enter sepal & petal dimensions to predict the Iris flower species.")

# User Inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width  = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width  = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict Species"):
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict
    pred_probs = model.predict(input_scaled)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    confidence = np.max(pred_probs) * 100
    
    st.success(f"🌺 Predicted Species: **{species_labels[pred_class]}**")
    st.info(f"Prediction Confidence: {confidence:.2f}%")

