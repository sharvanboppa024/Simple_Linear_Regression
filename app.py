
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

with open("SLR_MODEL.pkl", "rb") as file:
    model = pickle.load(file)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        experience = float(request.form["experience"])
        result = model.predict(np.array([[experience]]))[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted Salary: ₹{result:,.2f}",
            experience=experience
        )
    except (ValueError, TypeError):
        return render_template(
            "index.html",
            error="Please enter a valid number."
        )

if __name__ == "__main__":
    app.run(debug=True)
