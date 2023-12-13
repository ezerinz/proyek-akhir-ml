from flask import Flask, request, render_template
import pickle
import pandas as pd
import locale

app = Flask(__name__)
model = pickle.load(open("model/model.pkl", "rb"))
locale.setlocale(locale.LC_ALL, "id_ID")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.form
    pred_param = {
        "danceability_%": [data["danceability"]],
        "valence_%": [data["valence"]],
        "energy_%": [data["energy"]],
        "acousticness_%": [data["acousticness"]],
        "instrumentalness_%": [data["instrumental"]],
        "liveness_%": [data["liveness"]],
        "speechiness_%": [data["speechiness"]],
        "total_playlist": [data["total_playlist"]],
        "total_charts": [data["total_charts"]],
    }
    df = pd.DataFrame(pred_param)
    prediction = model.predict(df)

    output = prediction[0].item()
    output = locale.format_string("%0.2f", output, grouping=True)
    return render_template("prediksi.html", hasil=output)


@app.route("/")
def home():
    return render_template("index.html")
