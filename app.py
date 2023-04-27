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

def add_dummies_to_new_data(dummies_list, dummies_pickle_file, new_data_df):
    # Charger le dummies à partir du fichier pickle
    with open(dummies_pickle_file, 'rb') as handle:
        dummies = pickle.load(handle)
        
    # Créer un dataframe contenant uniquement les colonnes présentes dans le dummies list
    new_data_dummies = pd.DataFrame(columns=dummies_list)
    
    # Ajouter les dummies pour chaque ligne du nouveau dataframe
    for index, row in new_data_df.iterrows():
        categories = row['category'].split(',')
        for category in categories:
            if category not in dummies_list:
                dummies.loc[dummies.index[-1] + 1] = [0] * len(dummies.columns)
                dummies.loc[dummies.index[-1], 'category_other_category'] = 1
                dummies_list.append(category)
        new_data_dummies_row = [1 if category in categories else 0 for category in dummies_list]
        new_data_dummies.loc[index] = new_data_dummies_row

    # Concaténer les dummies à la nouvelle ligne de données
    new_data = pd.concat([new_data_df, new_data_dummies], axis=1)

    return new_data

