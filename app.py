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

def add_dummies_to_new_data(dummies_list, new_data_df):
    # Charger les dummies à partir du fichier pickle
    with open('dummies.pickle', 'rb') as handle:
        dummies = pickle.load(handle)

    # Obtenir les catégories de la nouvelle ligne
    categories = new_data_df['category'].str.split(',', expand=True).values.flatten()

    # Vérifier si toutes les catégories sont présentes dans la liste de dummies
    if set(categories).issubset(set(dummies_list)):
        # Appliquer les dummies sur la nouvelle ligne
        new_data_dummies = pd.get_dummies(categories, prefix='category')
        new_data_dummies = new_data_dummies.reindex(columns=dummies_list, fill_value=0)
    else:
        # Trouver les catégories absentes du dummies
        missing_categories = set(categories) - set(dummies_list)

        # Ajouter les catégories absentes à category_other_category
        new_data_dummies = pd.DataFrame(columns=dummies_list)
        new_data_dummies['category_other_category'] = 0
        for category in missing_categories:
            new_data_dummies['category_other_category'] = new_data_dummies['category_other_category'] | (categories == category)

    # Concaténer les dummies à la nouvelle ligne de données
    new_data = pd.concat([new_data_df, new_data_dummies], axis=1)

    return new_data



