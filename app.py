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
['a', 'ai', 'aie', 'aient', 'aies', 'ait', 'alors', 'après', 'as', 'au', 'aucun', 'aura', 'aurai', 'auraient', 'aurais', 'aurait', 'auras', 'aurez', 'auriez', 'aurions', 'aurons', 'auront', 'aussi', 'autre', 'aux', 'avaient', 'avais', 'avait', 'avant', 'avec', 'avez', 'aviez', 'avions', 'avons', 'ayant', 'ayez', 'ayons', 'bien', 'bon', 'car', 'ce', 'cela', 'ces', 'cet', 'cette', 'ceux', 'chaque', 'ci', 'combien', 'comme', 'comment', 'd', 'dans', 'de', 'debout', 'dedans', 'dehors', 'delà', 'depuis', 'derrière', 'des', 'désormais', 'desquelles', 'desquels', 'dessous', 'dessus', 'devant', 'devers', 'devra', 'devrait', 'devront', 'dire', 'dois', 'doit', 'donc', 'dont', 'douze', 'douzième', 'dr', 'du', 'duquel', 'durant', 'dès', 'début', 'désormais', 'eh', 'elle', 'elles', 'en', 'encore', 'entre', 'envers', 'es', 'est', 'et', 'etc', 'etre', 'eu', 'eue', 'eues', 'euh', 'eurent', 'eus', 'eusse', 'eussent', 'eusses', 'eussiez', 'eussions', 'eut', 'eux', 'excepté', 'hormis', 'hors', 'huit', 'huitième', 'ici', 'il', 'ils', 'j', 'je', 'jusqu', 'jusque', 'l', 'la', 'laquelle', 'le', 'lequel', 'les', 'lesquelles', 'lesquels', 'leur', 'leurs', 'longtemps', 'lorsque', 'lui', 'là', 'lès', 'ma', 'maint', 'maintenant', 'mais', 'me', 'mes', 'mine', 'moi', 'moins', 'mon', 'mot', 'même', 'mêmes', 'n', 'ne', 'ni', 'non', 'nos', 'notre', 'nous', 'nul', 'néanmoins', 'nôtre', 'nôtres', 'on', 'ont', 'onze', 'onzième', 'or', 'ou', 'où', 'par', 'parce', 'parmi', 'partant', 'pas', 'passé', 'pendant', 'personne', 'peu', 'plus', 'plutôt', 'possible', 'pour', 'pourquoi', 'premier', 'première', 'premièrement', 'près', 'proche', 'ps', 'puisque', 'put', 'pût', 'qu', 'quand', 'quant', 'quatorze', 'quatrième', 'quatrièmement

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from keras.layers import Input, Embedding, Conv1D, MaxPooling1D, Flatten, Dense, concatenate
from keras.callbacks import EarlyStopping
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Charger les données dans un DataFrame pandas
df = pd.read_csv("chemin/vers/votre/dataset.csv")

# Combiner les colonnes "subject" et "attachment" en une seule colonne "text"
df['text'] = df['subject'].astype(str) + ' ' + df['attachment'].astype(str)

# Enlever les mots malveillants de la colonne "text"
malicious_words = ['spam', 'phishing', 'scam', 'fraud']
remove_malicious_words = lambda s: ' '.join([word for word in s.split() if word.lower() not in malicious_words])
df['text'] = df['text'].apply(remove_malicious_words)

# Diviser les données en ensemble d'entraînement et ensemble de test
X_train_text, X_test_text, X_train_mwc, X_test_mwc, y_train, y_test = train_test_split(df['text'], df['malicious_words_count'], df['target'], test_size=0.2, random_state=42)

# Convertir les textes en séquences de nombres
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(X_train_text)
X_train_text = tokenizer.texts_to_sequences(X_train_text)
X_test_text = tokenizer.texts_to_sequences(X_test_text)

# Remplir les séquences avec des zéros pour qu'elles aient toutes la même longueur
maxlen = 100
X_train_text = pad_sequences(X_train_text, padding='post', maxlen=maxlen)
X_test_text = pad_sequences(X_test_text, padding='post', maxlen=maxlen)

# Construire les couches d'entrée pour les données textuelles et les données de comptage de mots malveillants
text_input = Input(shape=(maxlen,), name='text_input')
mwc_input = Input(shape=(1,), name='mwc_input')

# Ajouter une couche d'embedding pour les données textuelles
emb = Embedding(10000, 128, input_length=maxlen)(text_input)

# Ajouter une couche de convolution pour les données textuelles
conv = Conv1D(64, 5, activation='relu')(emb)
pool = MaxPooling1D(pool_size=4)(conv)
flat = Flatten()(pool)

# Concaténer les sorties des couches d'embedding et de comptage de mots malveillants
concat = concatenate([flat, mwc_input], axis=-1)

# Ajouter une couche dense pour la sortie
out = Dense(1, activation='sigmoid')(concat)

# Construire le modèle
model = Model(inputs=[text_input, mwc_input], outputs=out)

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Entraîner le modèle
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
# Entraîner le modèle
early_stop = EarlyStopping(monitor='val_loss', patience=3, verbose=1)
model.fit({'text_input': X_train_text, 'mwc_input': X_train_mwc}, y_train,
          validation_data=({'text_input': X_test_text, 'mwc_input': X_test_mwc}, y_test),
          epochs=10, batch_size=32, callbacks=[early_stop])

# Prédire les étiquettes pour l'ensemble de test
y_pred = model.predict({'text_input': X_test_text, 'mwc_input': X_test_mwc})
y_pred = (y_pred > 0.5).astype(int)

# Calculer les métriques de classification
print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Precision : ", precision_score(y_test, y_pred))
print("Recall : ", recall_score(y_test, y_pred))
print("F1 Score : ", f1_score(y_test, y_pred))

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
tokenizer = Tokenizer(num_words=5000, lower=True)
tokenizer.fit_on_texts(X_train_text)
X_train_text_seq = tokenizer.texts_to_sequences(X_train_text)
X_test_text_seq = tokenizer.texts_to_sequences(X_test_text)
X_train_text_pad = pad_sequences(X_train_text_seq, maxlen=100)
X_test_text_pad = pad_sequences(X_test_text_seq, maxlen=100)
model = Sequential()
model.add(Embedding(5000, 32, input_length=100))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
early_stop = EarlyStopping(monitor='val_loss', patience=3)
model.fit([X_train_text_pad, X_train_mwc_norm], y_train, epochs=10, batch_size=32, validation_split=0.2, callbacks=[early_stop])

model.evaluate([X_test_text_pad, X_test_mwc_norm], y_test)
