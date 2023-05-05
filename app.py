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

def add_dummies_to_new_data(new_data_df):
    with open('app/model/data/ML0004_dummies_category_ARO.pkl', 'rb') as handle:
        dummies = pickle.load(handle)
    
    dummies_list = dummies.columns.tolist()

    new_data_dummies = new_data_df["category"].str.get_dummies(sep=',').add_prefix('category_')
    #new_data_dummies = new_data_dummies.reindex(columns=dummies_list, fill_value=0)
    new_data_dummies['category_other_category'] = 0

    for category in new_data_dummies:
        if category not in dummies.columns:
            new_data_df['category_other_category'] = 1
        """for category in missing_categories:
            new_data_dummies['category_other_category'] = new_data_dummies['category_other_category'] | new_data_dummies[category]
            del new_data_dummies[category]"""

    new_data = pd.concat([new_data_df, new_data_dummies], axis=1)

    return new_data


stopwords = set([
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "aren't", "as", "at", "be",
    "because", "been", "before", "being", "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during", "each", "few", "for", "from",
    "further", "had", "hadn't", "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in",
    "into", "is", "isn't", "it", "it's", "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out",
    "over", "own", "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these",
    "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when",
    "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves",
    "a", "ai", "aie", "aient", "aies", "ainsi", "ait", "allaient", "allo", "allons", "allô", "alors", "anterieur",
    "anterieure", "anterieures", "apres", "après", "as", "assez", "attendu", "au", "aucun", "aucune", "aucuns",
    "aujourd", "aujourd'hui", "aupres", "auquel", "aura", "aurai", "auraient", "aurais", "aurait", "auras", "aurez",
    "auriez", "aurions", "aurons", "auront", "aussi", "autre", "autrefois", "autrement", "autres", "autrui", "aux",
    "auxquelles", "auxquels", "avaient", "avais", "avait", "avant", "avec", "avoir", "avons", "ayant", "b", "bah",
    "bas", "basee", "bat", "beau", "beaucoup", "bien", "bigre", "bon", "boum", "bravo", "brrr", "c", "car", "ce",
    "ceci", "cela", "celle", "celle-ci", "celle-là", "celles", "celles-ci", "celles-là", "celui", "celui-ci",
    "celui-là", "celà", "cent", "cependant", "certain", "certaine", "certaines", "certains", "certes", "ces",
    "cet", "cette", "ceux", "ceux-ci", "ceux-là", "chacun", "chacune", "chaque", "cher", "chers", "chez", "chiche",
    "chut", "ci", "cinq", "cinquantaine", "cinquante", "cinquantième", "cinquième", "clac", "clic", "combien",
    "comme", "comment", "comparable", "comparables", "compris", "concernant", "contre", "couic", "crac", "d", "da",
    "dans", "de", "debout", "dedans", "dehors", "deja", "delà", "depuis", "dernier", "derniere", "derriere",
    "derrière", "des", "desormais", "desquelles", "desquels", "dessous", "dessus", "deux", "deuxième", "deuxièmement",
    "devant", "devers", "devra", "devrait", "different", "differentes", "differents", "différent", "différente",
    "différentes", "différents", "dire", "directe", "directement", "dit", "dite", "dits", "divers", "diverse",
    "diverses", "dix", "dix-huit", "dix-neuf", "dix-sept", "dixième", "doit", "doivent", "donc"
)

# 


