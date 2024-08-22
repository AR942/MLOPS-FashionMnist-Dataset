from flask import Flask, request, jsonify
import tensorflow as tf
from flask import render_template
import os
"""remove tensorflow & keras warning informations"""
import logging
import os
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #or {'0', '1', '2'}
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

model_path = "./model/model.h5"

model = tf.keras.models.load_model(model_path)
'_time', 'user', 'dhost', 'sum_bytes_in', 'sum_bytes_out',
       'avg_bytes_in', 'avg_bytes_out', 'min_bytes_in', 'min_bytes_out',
       'max_bytes_in', 'max_bytes_out', 'count', 'hour_of_day', 'day_of_week',
       'duration_since_last_query', 'duration_since_session_start', 'session',
       'bytes_ratio', 'request_rate', 'is_messagerie'


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/classify", methods=["POST"])
def classify():
    
    data = request.get_json()

   
    prediction = model.predict(data)

    
    return jsonify({"prediction": prediction.tolist()})



if __name__ == "__main__":
    app.run(host='0.0.0.0',port=1637, debug=True)
    
    
