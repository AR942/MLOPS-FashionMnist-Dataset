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


from sklearn.model_selection import train_test_split

# Diviser les données en ensembles d'entraînement et de test
X_train_text, X_test_text, X_train_mwc, X_test_mwc, y_train, y_test = train_test_split(
    text_data, mwc_data, labels, test_size=0.2, random_state=42)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Input, Embedding, SpatialDropout1D, LSTM, Conv1D, MaxPooling1D, Flatten, concatenate, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Convertir les textes en séquences de nombres
tokenizer = Tokenizer(num_words=1000)
tokenizer.fit_on_texts(X_train_text)
X_train_text = tokenizer.texts_to_sequences(X_train_text)
X_test_text = tokenizer.texts_to_sequences(X_test_text)

maxlen = 100
X_train_text = pad_sequences(X_train_text, padding='post', maxlen=maxlen)
X_test_text = pad_sequences(X_test_text, padding='post', maxlen=maxlen)

text_input = Input(shape=(maxlen,), name='text_input')
mwc_input = Input(shape=(31,), name='mwc_input')

emb = Embedding(1000, 128, input_length=maxlen)(text_input)
emb = SpatialDropout1D(0.2)(emb)
lstm = LSTM(64,dropout=0.2, recurrent_dropout=0.2, return_sequences=True)(emb)
conv = Conv1D(64, 5, activation='relu')(lstm)
pool = MaxPooling1D(pool_size=4)(conv)
flat = Flatten()(pool)

from tensorflow import keras
prec = keras.metrics.Precision()
rec = keras.metrics.Recall()

concat = concatenate([flat, mwc_input], axis=-1)

dense1 = Dense(64, activation='relu')(concat)
dense1 = Dropout(0.2)(dense1)
dense2 = Dense(32, activation='relu')(dense1)
out = Dense(1, activation='sigmoid')(dense2)

model = Model(inputs=[text_input, mwc_input], outputs=out)

model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.0005), metrics=[rec,prec])

#early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)
model.fit({'text_input': X_train_text, 'mwc_input': X_train_mwc}, y_train,
          validation_data=({'text_input': X_test_text, 'mwc_input': X_test_mwc}, y_test),
          epochs=50, batch_size=64, #callbacks=[early_stop]
         )

y_pred = model.predict({'text_input': X_test_text, 'mwc_input': X_test_mwc})
y_pred = (y_pred > 0.5).astype(int)

print("Accuracy : ", accuracy_score(y_test, y_pred))
print("Precision : ", precision_score(y_test, y_pred))
print("Recall : ", recall_score(y_test, y_pred))
print("F1 Score : ", f1_score(y_test, y_pred))

[0.70175439 0.71929825 0.65517241 0.70689655 0.61403509]
[0.90909091 1.         0.97435897 1.         0.97222222]
0.6794313369630974
0.9711344211344212
0.03926115564400642
0.03324882472475661


{'accuracy': 0.99, 'precision': 0.72, 'recall': 0.99}

{'accuracy': 0.99, 'precision': 0.67, 'recall': 0.95}


def find_similar_words(all_words):
    similar_words = []
    for word1 in all_words:
        similar_group = [word1]
        for word2 in all_words:
            if are_similar_words(word1, word2):
                similar_group.append(word2)
        similar_words.append(similar_group)
    similar_words = [group for group in similar_words if len(group) > 1]
    return similar_words


def are_similar_words(word1, word2):
    cleaned_word1 = ''.join(c.lower() for c in word1 if c.isalpha())
    cleaned_word2 = ''.join(c.lower() for c in word2 if c.isalpha())
    return cleaned_word1.startswith(cleaned_word2) or cleaned_word2.startswith(cleaned_word1)


# Appliquer les fonctions pour obtenir les mots similaires dans la colonne texte
all_words = get_unique_words(df['text'])
similar_words = find_similar_words(all_words)

import pandas as pd
import re

# Liste des mots à détecter
mots_detecter = ['CV', 'Candidature', 'Proposition']

# Exemple de données
data = pd.DataFrame({'attachment': ['CV John Doe.pdf', 'Candidature_Smith.docx', 'Contrat BP 2023.pdf', 'Leasing Agreement.doc', 'bpli.txt']})

# Vérification des mots dans la colonne "attachment"
regex = r'(?i)(?:\b(?:{})\b.*){{2,}}'.format('|'.join(mots_detecter))
data['presence_mots'] = data['attachment'].str.contains(regex, regex=True)

print(data)


import pandas as pd

# Groupes de mots à détecter
groupes = [
    ['candidat', 'CV'],
    ['candidat', 'proposition'],
    ['candidat', 'internship'],
    ['CV', 'internship'],
    ['CV', 'stage']
]

# Exemple de données
data = pd.DataFrame({
    'subject': ['CV pour candidat', 'Offre de proposition', 'Stage pour candidat', 'CV et stage'],
    'attachment': ['CV_John_Doe.pdf', 'CV_Candidate.docx', 'Proposition.pdf', 'CV_Internship.doc']
})

# Vérification des groupes de mots dans les colonnes "subject" et "attachment"
for groupe in groupes:
    mot1, mot2 = groupe
    conditions = [
        (data['subject'].str.contains(fr'\b{mot1}\b', case=False) & data['attachment'].str.contains(fr'\b{mot2}\b', case=False)),
        (data['subject'].str.contains(fr'\b{mot2}\b', case=False) & data['attachment'].str.contains(fr'\b{mot1}\b', case=False)),
        (data['subject'].str.contains(fr'\b{mot1}\b', case=False) & ~data['attachment'].str.contains(fr'\b{mot2}\b', case=False)),
        (data['attachment'].str.contains(fr'\b{mot1}\b', case=False) & ~data['subject'].str.contains(fr'\b{mot2}\b', case=False))
    ]
    data[f'groupe_{mot1}_{mot2}'] = pd.Series(conditions).any()

print(data)

import pandas as pd

# Groupes de mots à détecter
groupes = [
    ['candidat', 'CV'],
    ['candidat', 'proposition'],
    ['candidat', 'internship'],
    ['CV', 'internship'],
    ['CV', 'stage']
]

# Exemple de données
data = pd.DataFrame({
    'subject': ['CV pour candidat', 'Offre de proposition', 'Stage pour candidat', 'CV et stage'],
    'attachment': ['CV_John_Doe.pdf', 'CV_Candidate.docx', 'Proposition.pdf', 'CV_Internship.doc']
})

# Concaténation des colonnes "subject" et "attachment"
data['texte_concatene'] = data['subject'] + ' ' + data['attachment']

# Vérification des groupes de mots dans la colonne "texte_concatene"
for groupe in groupes:
    mot1, mot2 = groupe
    condition = data['texte_concatene'].str.contains(fr'\b{mot1}\b', case=False) & data['texte_concatene'].str.contains(fr'\b{mot2}\b', case=False)
    data[f'groupe_{mot1}_{mot2}'] = condition

print(data)
