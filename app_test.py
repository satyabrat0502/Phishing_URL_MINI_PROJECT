from flask import Flask, request, render_template
import numpy as np
import pickle
import warnings
import os
from feature import FeatureExtraction  # Ensure this module is available

warnings.filterwarnings("ignore")

# Load the trained model
model_path = os.path.join(os.path.dirname(__file__), "pickle", "model.pkl")

try:
    with open(model_path, "rb") as file:
        gbc = pickle.load(file)
except Exception as e:
    print(f"Error loading model: {e}")
    gbc = None  # Set model to None if loading fails

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        url = request.form["url"]
        obj = FeatureExtraction(url)
        x = np.array(obj.getFeaturesList()).reshape(1, -1)  # Ensure correct shape

        if gbc is None:
            return render_template("index.html", error="Model failed to load.")

        try:
            y_pred = gbc.predict(x)[0]
            y_pro_phishing = gbc.predict_proba(x)[0, 0]
            y_pro_non_phishing = gbc.predict_proba(x)[0, 1]
            pred = f"It is {y_pro_phishing * 100:.2f}% safe to go."
            return render_template(
                "index.html", xx=round(y_pro_non_phishing, 2), url=url
            )
        except Exception as e:
            return render_template("index.html", error=f"Prediction error: {e}")

    return render_template("index.html", xx=-1)


if __name__ == "__main__":
    app.run(debug=True)
