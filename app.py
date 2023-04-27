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

def add_dummies_to_new_data(dummies, new_data_df):
    # Initialiser une série de colonnes category_other_category à 0
    other_category_cols = pd.Series(0, index=dummies.columns[dummies.columns.str.startswith('category_')])

    # Parcourir chaque ligne du nouveau DataFrame
    for i, row in new_data_df.iterrows():
        # Extraire les catégories de la ligne
        categories = row['category'].split(',')

        # Vérifier si chaque catégorie est présente dans les dummies
        for category in categories:
            if category not in dummies.columns:
                other_category_cols['category_other_category'] = 1
                break

        # Concaténer les dummies et les colonnes category_other_category à la ligne de données
        new_data_df.loc[i, dummies.columns] = 0  # Initialiser les colonnes à 0
        new_data_df.loc[i, other_category_cols.index] = other_category_cols.values
        for category in categories:
            if category in dummies.columns:
                new_data_df.loc[i, category] = 1

    return new_data_df


