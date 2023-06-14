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

import pandas as pd

# Convertir la colonne "_time" en format de date et heure
df[time_col] = pd.to_datetime(df[time_col], format="%Y-%m-%d %H:%M:%S.%f%z")

# Créer une fonctionnalité pour représenter l'heure de la journée
df['hour_of_day'] = df[time_col].dt.hour

# Créer une fonctionnalité pour représenter le jour de la semaine
df['day_of_week'] = df[time_col].dt.dayofweek

# Trier le DataFrame par utilisateur et horodatage
df.sort_values(by=['user', time_col], inplace=True)

# Créer une fonctionnalité pour représenter les sessions de chaque utilisateur
df['session'] = (df['user'] != df['user'].shift()).cumsum()

# Créer une fonctionnalité pour représenter le ratio de bytes
df['bytes_ratio'] = df['sumbytesin'] / df['sumbytesout']

# Créer une fonctionnalité pour représenter le taux de requêtes
df['request_rate'] = df['countofrequests'] / (df[time_col] - df.groupby('user')[time_col].transform('first')).dt.total_seconds()

# Créer une fonctionnalité pour représenter la durée depuis la première requête
df['duration_since_first_request'] = (df[time_col] - df.groupby('user')[time_col].transform('first')).dt.total_seconds()

# Créer une fonctionnalité pour représenter la durée depuis la dernière requête
df['duration_since_last_request'] = (df[time_col] - df.groupby('user')[time_col].shift()).dt.total_seconds()

# Supprimer les colonnes inutiles
df.drop(columns=[time_col], inplace=True)

# Afficher le DataFrame mis à jour
print(df.head())


import pandas as pd

# Calcul de la longueur des dhost
df['dhost_length'] = df['dhost'].apply(lambda x: len(x))

# Comptage du nombre de points dans les dhost
df['dhost_num_dots'] = df['dhost'].apply(lambda x: x.count('.'))

# Nombre d'occurrences d'un hôte spécifique
df['host_occurrences'] = df.groupby('dhost')['dhost'].transform('count')

# Nombre d'hôtes uniques visités par utilisateur
df['unique_hosts_per_user'] = df.groupby('user')['dhost'].transform('nunique')

# Fréquence des interactions avec un hôte spécifique
df['host_interaction_frequency'] = df.groupby(['user', 'dhost'])['dhost'].transform('count') / df.groupby('user')['dhost'].transform('count')

# Indicateur de présence d'un hôte spécifique
hosts_to_check = ['example.com', 'google.com', 'yahoo.com']
for host in hosts_to_check:
    df[f'host_{host}_presence'] = df['dhost'].apply(lambda x: 1 if host in x else 0)

# Longueur moyenne des noms d'hôtes
df['average_host_length'] = df.groupby('user')['dhost_length'].transform('mean')

# Nombre de sous-domaines
df['subdomain_count'] = df['dhost'].apply(lambda x: x.count('.') + 1)

# Occurrence de certains motifs dans les noms d'hôtes
keywords_to_check = ['phishing', 'malware', 'bank']
for keyword in keywords_to_check:
    df[f'keyword_{keyword}_presence'] = df['dhost'].apply(lambda x: 1 if keyword in x else 0)

# Affichage du DataFrame avec les nouvelles caractéristiques


import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Sélection des colonnes à utiliser pour l'autoencodeur
columns_to_encode = ['dhost_length', 'dhost_num_dots', 'host_occurrences', 'unique_hosts_per_user',
                     'host_interaction_frequency', 'average_host_length', 'subdomain_count']

# Sélection des données pour l'autoencodeur
data = df_[columns_to_encode].values

# Standardisation des données
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# Division des données en ensembles d'entraînement et de test
X_train, X_test = train_test_split(data_scaled, test_size=0.2, random_state=42)

# Construction de l'architecture de l'autoencodeur
input_dim = X_train.shape[1]
encoding_dim = 4

input_layer = tf.keras.Input(shape=(input_dim,))
encoder = tf.keras.layers.Dense(encoding_dim, activation='relu')(input_layer)
decoder = tf.keras.layers.Dense(input_dim, activation='sigmoid')(encoder)
autoencoder = tf.keras.Model(inputs=input_layer, outputs=decoder)

# Compilation du modèle de l'autoencodeur
autoencoder.compile(optimizer='adam', loss='mse')

# Entraînement de l'autoencodeur
autoencoder.fit(X_train, X_train, epochs=10, batch_size=32, shuffle=True, validation_data=(X_test, X_test))

# Reconstruction des données avec l'autoencodeur
reconstructed_data = autoencoder.predict(data_scaled)

# Calcul de l'erreur de reconstruction
mse = np.mean(np.power(data_scaled - reconstructed_data, 2), axis=1)

# Ajout de l'erreur de reconstruction au DataFrame
df_['reconstruction_error'] = mse

# Détection des cas anormaux
threshold = np.percentile(df_['reconstruction_error'], 95)
df_['is_anomaly'] = df_['reconstruction_error'].apply(lambda x: 1 if x > threshold else 0)

---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
File /usr/local/lib/python3.9/site-packages/pandas/core/indexes/base.py:3802, in Index.get_loc(self, key, method, tolerance)
   3801 try:
-> 3802     return self._engine.get_loc(casted_key)
   3803 except KeyError as err:

File /usr/local/lib/python3.9/site-packages/pandas/_libs/index.pyx:138, in pandas._libs.index.IndexEngine.get_loc()

File /usr/local/lib/python3.9/site-packages/pandas/_libs/index.pyx:165, in pandas._libs.index.IndexEngine.get_loc()

File pandas/_libs/hashtable_class_helper.pxi:5745, in pandas._libs.hashtable.PyObjectHashTable.get_item()

File pandas/_libs/hashtable_class_helper.pxi:5753, in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 0

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
Cell In[16], line 7
      5 distances = np.zeros(len(df_embed))
      6 for i in range(len(df_embed)):
----> 7     distances[i] = min([euclidean(df_embed[i], df_mebed[j]) for j in indices_target_1])
     10 seuil = 0.1
     12 indices_point_proches = np.where(distances <= seuil)[0]

Cell In[16], line 7, in <listcomp>(.0)
      5 distances = np.zeros(len(df_embed))
      6 for i in range(len(df_embed)):
----> 7     distances[i] = min([euclidean(df_embed[i], df_mebed[j]) for j in indices_target_1])
     10 seuil = 0.1
     12 indices_point_proches = np.where(distances <= seuil)[0]

File /usr/local/lib/python3.9/site-packages/pandas/core/frame.py:3807, in DataFrame.__getitem__(self, key)
   3805 if self.columns.nlevels > 1:
   3806     return self._getitem_multilevel(key)
-> 3807 indexer = self.columns.get_loc(key)
   3808 if is_integer(indexer):
   3809     indexer = [indexer]

File /usr/local/lib/python3.9/site-packages/pandas/core/indexes/base.py:3804, in Index.get_loc(self, key, method, tolerance)
   3802     return self._engine.get_loc(casted_key)
   3803 except KeyError as err:
-> 3804     raise KeyError(key) from err
   3805 except TypeError:
   3806     # If we have a listlike key, _check_indexing_error will raise
   3807     #  InvalidIndexError. Otherwise we fall through and re-raise
   3808     #  the TypeError.
   3809     self._check_indexing_error(key)

KeyError: 0


'_time', 'user', 'dhost', 'sum_bytes_in', 'sum_bytes_out',
       'avg_bytes_in', 'avg_bytes_out', 'min_bytes_in', 'min_bytes_out',
       'max_bytes_in', 'max_bytes_out', 'count'
       
       
       import pandas as pd
import numpy as np

def create_features(df):
    # 3. Statistiques des octets
    df['diff_bytes_in'] = df['max_bytes_in'] - df['min_bytes_in']
    df['diff_bytes_out'] = df['max_bytes_out'] - df['min_bytes_out']
    df['sum_bytes_total'] = df['sum_bytes_in'] + df['sum_bytes_out']
    df['avg_bytes_total'] = (df['avg_bytes_in'] + df['avg_bytes_out']) / 2

    # 4. Fréquence des requêtes
    df['requests_per_hour'] = df['count'] / 1  # Modifier ici la période de temps si nécessaire

    # 7. Tendance des requêtes
    df['_time'] = pd.to_datetime(df['_time'])  # Conversion en type de données datetime si nécessaire
    df['timestamp'] = df['_time'].astype(int) / 10**9  # Conversion en timestamp
    df['linear_regression_count'] = df.groupby('user')['count'].transform(lambda x: np.polyfit(df['timestamp'], x, 1)[0])

    # Calculer le nombre total de requêtes pour chaque utilisateur
    df['total_requests_per_user'] = df.groupby('user')['count'].transform('sum')

    return df


 length = len(df['user'])
    columns = ['_time', 'user', 'dhost', 'sum_bytes_in', 'sum_bytes_out', 'avg_bytes_in', 'avg_bytes_out',
               'min_bytes_in', 'min_bytes_out', 'max_bytes_in', 'max_bytes_out', 'count']
    for col in columns:
        if len(df[col]) != length:
            raise ValueError("La longueur de la colonne '{}' ne correspond pas à celle des autres colonnes.".format(col))
       
      
import pandas as pd

def extract_user_behavior_features(df):
    # Nombre de requêtes par session
    df['requests_per_session'] = df.groupby('user')['count'].transform('sum')
    
    # Durée de la session
    df['session_duration'] = df.groupby('user')['timestamp'].transform(lambda x: x.max() - x.min())
    
    # Heures d'activité
    df['hour_of_day'] = df['timestamp'].dt.hour
    df['active_in_morning'] = (df['hour_of_day'] >= 6) & (df['hour_of_day'] < 12)
    df['active_in_afternoon'] = (df['hour_of_day'] >= 12) & (df['hour_of_day'] < 18)
    df['active_in_evening'] = (df['hour_of_day'] >= 18) & (df['hour_of_day'] < 24)
    
    # Variation de comportement
    df['sum_bytes_in_diff'] = df.groupby('user')['sum_bytes_in'].transform(lambda x: x.diff().shift(-1))
    df['sum_bytes_out_diff'] = df.groupby('user')['sum_bytes_out'].transform(lambda x: x.diff().shift(-1))
    df['avg_bytes_in_diff'] = df.groupby('user')['avg_bytes_in'].transform(lambda x: x.diff().shift(-1))
    df['avg_bytes_out_diff'] = df.groupby('user')['avg_bytes_out'].transform(lambda x: x.diff().shift(-1))
    
    # Activité inter-session
    df['inter_session_time'] = df.groupby('user')['timestamp'].transform(lambda x: x.diff().shift(-1))
    
    return df

# Utilisation de la fonction pour extraire les fonctionnalités de comportement utilisateur
df_features = extract_user_behavior_features(df)


user_features = df_filtered.groupby('user').agg({
    'count': ['mean', 'std'],
    'sum_bytes_in': ['mean', 'std'],
    'sum_bytes_out': ['mean', 'std'],
    'avg_bytes_in': ['mean', 'std'],
    'avg_bytes_out': ['mean', 'std'],
    'min_bytes_in': ['mean', 'std'],
    'min_bytes_out': ['mean', 'std'],
    'max_bytes_in': ['mean', 'std'],
    'max_bytes_out': ['mean', 'std'],
    'var_bytes_in': ['mean', 'std'],
    'var_bytes_out': ['mean', 'std'],
    'unique_dhosts': ['mean', 'std'],
    'diff_bytes_in': ['mean', 'std'],
    'diff_bytes_out': ['mean', 'std'],
    'sum_bytes_total': ['mean', 'std'],
    'avg_bytes_total': ['mean', 'std'],
    'requests_per_hour': ['mean', 'std'],
    'linear_regression_count': ['mean', 'std'],
    'total_requests_per_user': ['mean', 'std'],
    'requests_per_session': ['mean', 'std'],
    'session_duration': ['mean', 'std'],
    'hour_of_day': ['mean', 'std'],
    'active_in_morning': ['mean', 'std'],
    'active_in_afternoon': ['mean', 'std'],
    'active_in_evening': ['mean', 'std'],
    'sum_bytes_in_diff': ['mean', 'std'],
    'sum_bytes_out_diff': ['mean', 'std'],
    'avg_bytes_in_diff': ['mean', 'std'],
    'avg_bytes_out_diff': ['mean', 'std'],
    'inter_session_time': ['mean', 'std']
})

from sklearn.ensemble import IsolationForest

# Agrégation des caractéristiques par utilisateur
user_features = df.groupby('user').agg({
    'sum_bytes_in': 'sum',
    'sum_bytes_out': 'sum',
    'avg_bytes_in': 'mean',
    'avg_bytes_out': 'mean',
    # Autres caractéristiques agrégées
}).reset_index()

# Séparation des caractéristiques et de la cible
X = user_features.drop('user', axis=1)

# Création et entraînement du modèle Isolation Forest
model = IsolationForest(contamination=0.05)  # Définir le niveau de contamination en fonction de votre cas
model.fit(X)

# Prédiction des utilisateurs suspects
predictions = model.predict(X)

# Ajout des prédictions au DataFrame des caractéristiques des utilisateurs
user_features['anomaly'] = predictions

# Filtrage des utilisateurs suspects
suspect_users = user_features[user_features['anomaly'] == -1]

# Affichage des résultats
print(suspect_users)

# Normalisation des caractéristiques
scaler = StandardScaler()
user_features_scaled = scaler.fit_transform(user_features)

# Réduction de dimension avec PCA
pca = PCA(n_components=2)
user_features_pca = pca.fit_transform(user_features_scaled)

# Entraînement du modèle OneClassSVM
svm_model = OneClassSVM()
svm_model.fit(user_features_pca)

# Prédiction des utilisateurs suspects
user_features['anomaly_score'] = svm_model.decision_function(user_features_pca)
user_features['is_suspect'] = user_features['anomaly_score'] < 0

# Affichage des utilisateurs suspects
suspect_users = user_features[user_features['is_suspect']]
print(suspect_users)
from sklearn.ensemble import IsolationForest

# Sélection des colonnes pertinentes pour la détection des utilisateurs malveillants
selected_columns = ['sum_bytes_in', 'sum_bytes_out', 'avg_bytes_in', 'avg_bytes_out',
                    'count', 'diff_bytes_in', 'diff_bytes_out', 'sum_bytes_total',
                    'avg_bytes_total', 'sum_bytes_in_diff', 'sum_bytes_out_diff',
                    'avg_bytes_in_diff', 'avg_bytes_out_diff', 'total_requests_per_user',
                    'inter_session_time', 'hour_of_day', 'day_of_week', 'is_weekend',
                    'is_night', 'duration_since_last_query', 'duration_since_session_start',
                    'session', 'bytes_ratio', 'request_rate', 'is_messagerie']

# Extraction des caractéristiques pertinentes pour chaque utilisateur
user_features = df.groupby('user')[selected_columns].agg(['sum', 'mean', 'max', 'min'])
# Aplatir les colonnes multi-niveaux
user_features.columns = user_features.columns.to_flat_index()

# Aplatir les colonnes multi-niveaux
user_features.columns = ['_'.join(col) for col in user_features.columns]

# Entraînement du modèle Isolation Forest par utilisateur
model = IsolationForest(contamination=0.05)  # Définir le niveau de contamination en fonction de votre cas
model.fit(user_features)

# Prédiction des utilisateurs malveillants
predictions = model.predict(user_features)

# Filtrage des utilisateurs malveillants
malicious_users = user_features[predictions == -1].index

# Affichage des résultats
print(malicious_users)


'_time', 'user', 'dhost', 'sum_bytes_in', 'sum_bytes_out',
       'avg_bytes_in', 'avg_bytes_out', 'min_bytes_in', 'min_bytes_out',
       'max_bytes_in', 'max_bytes_out', 'count', 'diff_bytes_in',
       'diff_bytes_out', 'sum_bytes_total', 'avg_bytes_total',
       'sum_bytes_in_diff', 'sum_bytes_out_diff', 'avg_bytes_in_diff',
       'avg_bytes_out_diff', 'timestamp', 'linear_regression_count',
       'total_requests_per_user', 'inter_session_time', 'hour_of_day',
       'day_of_week', 'is_weekend', 'is_night', 'duration_since_last_query',
       'duration_since_session_start', 'session', 'bytes_ratio',
       'request_rate', 'is_messagerie'
