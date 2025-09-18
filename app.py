
import numpy as np
import gradio as gr
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler
import joblib

# Load trained model and scaler
model = load_model("iris_ann.h5")      # make sure you saved your ANN as iris_ann.h5
scaler = joblib.load("scaler.pkl")     # save your StandardScaler earlier

# Map encoded labels back to species names
species_labels = {0: "Iris-setosa", 1: "Iris-versicolor", 2: "Iris-virginica"}

# Define prediction function
def predict_species(sepal_length, sepal_width, petal_length, petal_width):
    # Create input array
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    
    # Scale input using the same scaler fitted before
    input_scaled = scaler.transform(input_data)
    
    # Get prediction
    pred_probs = model.predict(input_scaled)
    pred_class = np.argmax(pred_probs, axis=1)[0]
    
    return species_labels[pred_class]

# Gradio Interface
demo = gr.Interface(
    fn=predict_species,
    inputs=[
        gr.Number(label="Sepal Length (cm)"),
        gr.Number(label="Sepal Width (cm)"),
        gr.Number(label="Petal Length (cm)"),
        gr.Number(label="Petal Width (cm)")
    ],
    outputs=gr.Textbox(label="Predicted Species"),
    title="🌸 Iris Flower Classifier (ANN)",
    description="Enter sepal & petal dimensions to predict the Iris flower species using a trained ANN."
)

demo.launch()
