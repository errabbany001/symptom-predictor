from flask import Flask, request, jsonify, render_template
import numpy as np
import json
import pickle

# --- Load model parameters ---
with open("model_parametres.json", "r") as f:
    param_data = json.load(f)
parametres = {k: np.array(v) for k, v in param_data.items()}

# --- Load encoder ---
with open("encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

# --- Softmax and forward propagation ---
def softmax(x):
    expX = np.exp(x - np.max(x, axis=0, keepdims=True))
    return expX / expX.sum(axis=0, keepdims=True)

def forward_propagation(X, parametres):
    activations = {'A0': X}
    C = len(parametres) // 2
    for c in range(1, C + 1):
        Z = parametres['W' + str(c)] @ activations['A' + str(c - 1)] + parametres['b' + str(c)]
        A = softmax(Z) if c == C else 1 / (1 + np.exp(-Z))
        activations['A' + str(c)] = A
    return activations

# --- Prediction ---
def predict_single_disease(x_single, parametres, encoder):
    x_single = np.array(x_single).reshape(-1, 1)
    activations = forward_propagation(x_single, parametres)
    C = len(parametres) // 2
    Af = activations['A' + str(C)]
    predicted_index = np.argmax(Af, axis=0)[0]
    predicted_class = encoder.categories_[0][predicted_index]
    return predicted_class

# --- Flask app setup ---
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # expects templates/index.html to exist

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    x_input = data["features"]
    prediction = predict_single_disease(x_input, parametres, encoder)
    return jsonify({"prediction": prediction})

if __name__ == "__main__":
    app.run(debug=True)
