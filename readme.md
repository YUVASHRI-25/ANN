

# 🌸 Iris Flower Classifier (ANN + Gradio)

This project builds an **Artificial Neural Network (ANN)** to classify Iris flowers into three species (*Iris-setosa, Iris-versicolor, Iris-virginica*) based on their sepal and petal dimensions.
A **Gradio web app** is created to make the model interactive.

---

## 📂 Project Structure

```
│── Iris.csv              # Dataset
│── train_ann.py          # Script to train and save ANN + scaler
│── app.py                # Gradio app for interactive prediction
│── iris_ann.h5           # Saved ANN model
│── scaler.pkl            # Saved StandardScaler
│── README.md             # Project documentation
```

---

## ⚙️ Installation

1. Clone this repo or download the files.
2. Install dependencies:

   ```bash
   pip install numpy pandas scikit-learn tensorflow gradio joblib matplotlib seaborn
   ```
3. Make sure `iris_ann.h5` and `scaler.pkl` are in the same folder as `app.py`.

---

## 📊 Dataset

* **Source**: [Iris Dataset](https://archive.ics.uci.edu/ml/datasets/iris)
* **Features**:

  * SepalLengthCm
  * SepalWidthCm
  * PetalLengthCm
  * PetalWidthCm
* **Target**: Species (Setosa, Versicolor, Virginica)

---

## 🧠 Model (ANN)

* Framework: **TensorFlow Keras**
* Preprocessing: StandardScaler
* Architecture:

  * Input Layer: 4 features
  * Hidden Layer 1: Dense(8, activation="relu")
  * Hidden Layer 2: Dense(8, activation="relu")
  * Output Layer: Dense(3, activation="softmax")
* Loss: `sparse_categorical_crossentropy`
* Optimizer: `adam`
* Metrics: `accuracy`

---

## 🚀 Training

Run the training script:

```bash
python train_ann.py
```

This saves:

* `iris_ann.h5` → Trained ANN model
* `scaler.pkl` → StandardScaler

---

## 🌐 Gradio App

Run the app:

```bash
python app.py
```

This launches a local Gradio web app where you can input:

* Sepal Length
* Sepal Width
* Petal Length
* Petal Width

and get the **predicted Iris species**.


---

✨ Built with **TensorFlow, Scikit-learn, Gradio**

