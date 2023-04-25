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
    
    
import spacy
import pandas as pd

# Charger les modèles de langue anglaise et française
nlp_en = spacy.load('en_core_web_sm')
nlp_fr = spacy.load('fr_core_news_sm')

# Définir une fonction pour lemmatiser un texte en anglais ou en français en fonction de sa langue
def lemmatize_text(text):
    doc = nlp_en(text)
    if doc[0].lang_ == 'fr':
        doc = nlp_fr(text)
    return " ".join([token.lemma_ for token in doc])

# Définir la fonction pour traiter la colonne 'subject'
def process_subject(df):
    # Appliquer la fonction de lemmatisation à la colonne 'subject'
    df['subject'] = df['subject'].apply(lemmatize_text)
    
    # Supprimer les caractères spéciaux et les chiffres
    df['subject'] = df['subject'].str.replace('[^a-zA-Z]', ' ')
    
    # Mettre en minuscule
    df['subject'] = df['subject'].str.lower()
    
    # Supprimer les stopwords en anglais et en français
    spacy_stopwords_en = spacy.lang.en.stop_words.STOP_WORDS
    spacy_stopwords_fr = spacy.lang.fr.stop_words.STOP_WORDS
    stopwords = set().union(spacy_stopwords_en, spacy_stopwords_fr)
    df['subject'] = df['subject'].apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords]))
    
    # Supprimer les mots avec une longueur inférieure à 3
    df['subject'] = df['subject'].apply(lambda x: ' '.join([word for word in x.split() if len(word) > 2]))
    
    return df

from datetime import datetime
import pytz

def format_datetime(df, column):
    fmt = '%Y-%m-%dT%H:%M:%S.%f%z'
    tz_remote = pytz.timezone('UTC')
    tz_local = pytz.timezone('Europe/Paris')
    dt = column.apply(lambda x: datetime.strptime(x[:-6], fmt))
    dt_remote = dt.apply(lambda x: tz_remote.localize(x, is_dst=None))
    dt_local = dt_remote.apply(lambda x: x.astimezone(tz_local))
    return dt_local.dt.strftime('%Y-%m-%d %H:%M:%S')

import pandas as pd
import pytz

def format_time_col(time_col):
    # Convertir la colonne '_time' en un objet datetime avec le fuseau horaire d'origine
    dt = pd.to_datetime(time_col, format='%Y-%m%dT%H:%M:%S.%f%z').dt.tz_convert(pytz.UTC)
    # Convertir le temps au fuseau horaire de votre choix
    dt = dt.dt.tz_convert('Europe/Paris')
    # Formater la colonne '_time' au format souhaité
    return dt.dt.strftime('%Y-%m-%d %H:%M:%S')

# Charger les données dans un DataFrame
df = pd.read_csv('data.csv')

# Formater la colonne '_time'
df['_time'] = df['_time'].apply(format_time_col)

# Convertir la colonne '_time' en objet datetime
df['_time'] = pd.to_datetime(df['_time'], format='%Y-%m-%d %H:%M:%S')

# Extraire les informations de temps souhaitées
df['hour_of_day'] = df['_time'].dt.hour
df['day_of_week'] = df['_time'].dt.dayofweek
df['is_weekend'] = df['_time'].dt.dayofweek.isin([5,6]).astype(int)
df['is_night'] = ((df['_time'].dt.hour >= 22) | (df['_time'].dt.hour < 7)).astype(int)

