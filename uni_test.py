
"""remove tensorflow & keras warning informations"""
import logging
import os
logging.getLogger("tensorflow").setLevel(logging.WARNING)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  #or {'0', '1', '2'}
import warnings
warnings.filterwarnings("ignore")

import unittest
import requests
from app import app
Bonjour Philippe,

Au cours de mes deux stages de Master, j’ai eu l’opportunité de passer plusieurs mois au sein du SOC. J’ai grandement apprécié l’environnement de travail et j’ai pu développer des relations solides avec les membres de l’équipe. 
Je souhaite donc exprimer mon vif intérêt à rejoindre l’équipe en tant qu’interne à l’issue de mon stage. Je suis convaincu que mon engagement et mon expérience au SOC me permettront de contribuer de manière significative aux projets en cours et futurs.

Je suis conscient des défis et des responsabilités qui accompagnent ce rôle mais je suis prêt à m’investir pleinement et à travailler en étroite collaboration avec l’équipe pour atteindre nos objectifs communs.

Je reste à disposition pour discuter davantage de ma candidature et de mon projet. Je te remercie sincèrement pour ta considération et ton soutien continu.

Cordialement/Regards,

Bonjour Philippe,

Élève ingénieur en dernière année de Efrei Paris, j’ai pu réaliser 2 stages en 2002 et cette année 2023 dans le département…….

Le travail confié m’a permis de développer des qualités techniques professionnelles grâce notamment à la qualité les membres de l’équipe. 

Je manifeste un intérêt motivé à vous rejoindre en tant qu’interne à l’issue de mon stage. 

Je suis convaincu que mon engagement et mon expérience au SOC me permettront de contribuer à la création de valeur ajoutée.

Je suis préparé à relever les défis avec responsabilités pour atteindre les objectifs fixés.

Je vous soumets donc ma candidature en vous joignant mon CV et reste à votre disposition pour un entretien.

Je te remercie sincèrement pour ta considération et ton soutien constant.

Cordialement/Regards,
class FlaskTestCase(unittest.TestCase):

    def setUp(self):
        self.app = app.test_client()

    def test_home(self):
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)

    def test_classify(self):
        test_data = [[0, 0, 0, 0, 0, 0, 0, 9, 8, 0, 0, 34, 29, 7, 0, 11, 24, 0, 0, 3, 3, 1, 0, 1, 1, 0, 0, 0, 0, 0, 4, 0, 0, 1, 0, 0, 0, 0, 0, 44, 88, 99, 122, 123, 80, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0, 0, 3, 46, 174, 249, 67, 0, 94, 210, 61, 14, 212, 157, 37, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 2, 2, 0, 23, 168, 206, 242, 239, 238, 214, 125, 61, 113, 74, 133, 236, 238, 236, 203, 184, 20, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 175, 245, 223, 207, 205, 206, 216, 255, 237, 251, 232, 223, 212, 200, 205, 216, 249, 173, 0, 0, 2, 0, 0, 0, 0, 7, 0, 53, 225, 201, 197, 200, 201, 206, 199, 197, 185, 194, 204, 232, 226, 249, 219, 194, 205, 229, 33, 0, 1, 0, 0, 0, 0, 1, 0, 133, 223, 208, 192, 195, 233, 226, 216, 191, 210, 188, 236, 186, 0, 50, 234, 207, 208, 231, 133, 0, 0, 0, 0, 0, 0, 0, 0, 216, 218, 216, 194, 229, 172, 64, 219, 201, 200, 200, 247, 68, 72, 54, 165, 237, 212, 219, 226, 0, 0, 0, 0, 0, 0, 0, 50, 221, 207, 220, 211, 207, 165, 138, 205, 192, 191, 190, 232, 119, 113, 67, 173, 237, 217, 208, 221, 29, 0, 0, 0, 0, 0, 0, 131, 216, 200, 219, 207, 212, 231, 226, 193, 214, 224, 206, 203, 230, 122, 112, 234, 224, 214, 204, 224, 123, 0, 0, 0, 0, 0, 0, 195, 212, 204, 211, 203, 205, 200, 184, 213, 162, 138, 193, 207, 203, 231, 245, 208, 220, 211, 203, 219, 179, 0, 0, 0, 0, 0, 8, 185, 191, 218, 233, 219, 201, 221, 213, 246, 114, 127, 80, 129, 232, 198, 218, 207, 236, 227, 220, 216, 172, 21, 0, 0, 0, 0, 21, 4, 5, 64, 160, 224, 224, 144, 187, 197, 211, 207, 186, 192, 210, 212, 218, 225, 236, 177, 106, 56, 28, 1, 0, 0, 0, 0, 1, 1, 0, 2, 0, 116, 252, 96, 120, 51, 73, 70, 123, 79, 76, 64, 162, 252, 118, 1, 3, 0, 4, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 115, 226, 145, 170, 155, 165, 161, 159, 125, 175, 140, 174, 236, 95, 0, 2, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0, 131, 225, 204, 217, 221, 220, 217, 224, 231, 226, 237, 203, 237, 102, 0, 4, 2, 1, 2, 0, 0, 0, 0, 1, 1, 0, 3, 0, 135, 223, 201, 199, 194, 198, 195, 198, 192, 203, 199, 207, 231, 112, 0, 4, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 134, 223, 199, 206, 199, 201, 200, 203, 206, 207, 210, 206, 227, 119, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 139, 223, 198, 204, 200, 201, 200, 201, 204, 206, 208, 206, 229, 128, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 145, 223, 195, 205, 201, 201, 200, 204, 204, 206, 211, 205, 230, 139, 0, 2, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 157, 221, 194, 204, 204, 201, 201, 203, 205, 208, 211, 204, 230, 148, 0, 2, 0, 1, 1, 0, 0, 0, 0, 1, 1, 1, 0, 0, 166, 220, 194, 203, 203, 205, 203, 203, 206, 207, 212, 204, 230, 157, 0, 2, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 171, 221, 195, 206, 200, 199, 203, 203, 205, 206, 207, 204, 226, 181, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 165, 224, 197, 201, 208, 199, 204, 205, 207, 210, 213, 207, 229, 187, 0, 1, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 201, 203, 201, 207, 211, 203, 205, 206, 210, 213, 205, 225, 191, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 141, 201, 191, 188, 194, 187, 187, 191, 193, 195, 199, 199, 218, 161, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 212, 240, 213, 239, 233, 239, 231, 232, 236, 242, 245, 224, 245, 234, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 37, 69, 94, 123, 127, 138, 138, 142, 145, 135, 125, 103, 87, 56, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 209, 190, 181, 150, 170, 193, 180, 219, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24, 235, 210, 241, 222, 171, 220, 199, 236, 27, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 103, 227, 217, 218, 222, 189, 216, 201, 215, 103, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 197, 221, 201, 212, 215, 211, 215, 210, 228, 174, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 206, 202, 193, 202, 210, 209, 214, 193, 151, 132, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 201, 166, 202, 202, 219, 225, 208, 142, 110, 102, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 172, 136, 201, 226, 182, 213, 236, 146, 125, 73, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 43, 171, 147, 206, 253, 76, 176, 252, 158, 142, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 91, 201, 169, 200, 255, 0, 167, 255, 157, 176, 70, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 105, 211, 178, 204, 224, 0, 145, 225, 165, 186, 59, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 114, 209, 177, 212, 155, 0, 125, 221, 162, 197, 47, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 201, 180, 253, 46, 0, 116, 215, 155, 187, 31, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129, 179, 194, 226, 0, 0, 93, 214, 146, 171, 22, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 128, 168, 205, 168, 0, 0, 78, 217, 142, 154, 26, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 143, 168, 204, 181, 0, 0, 82, 223, 142, 136, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 180, 211, 140, 0, 0, 52, 221, 151, 124, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 187, 208, 182, 0, 0, 0, 212, 166, 131, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 60, 199, 208, 227, 0, 0, 12, 218, 176, 150, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 27, 234, 209, 211, 16, 0, 45, 229, 188, 157, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 243, 207, 218, 72, 0, 55, 232, 212, 183, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 213, 218, 224, 93, 0, 62, 236, 220, 175, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 168, 230, 228, 106, 0, 83, 237, 226, 149, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 132, 231, 229, 132, 0, 111, 241, 227, 146, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 93, 233, 234, 141, 0, 88, 242, 230, 153, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 29, 231, 235, 172, 0, 81, 242, 231, 145, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 227, 234, 186, 0, 92, 244, 240, 100, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 220, 244, 206, 0, 87, 248, 238, 80, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 233, 155, 0, 65, 235, 216, 34, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

        url = "http://192.168.1.13:1637/classify"
        response = requests.post(url, json=test_data)
        
        result = response.json()
        expected_value = {"prediction": [
        [
            1809.6632080078125,
            -390.2417907714844,
            57.099945068359375,
            -315.7286071777344,
            -548.5411987304688,
            -3536.708984375,
            1508.170166015625,
            -3511.53515625,
            -418.4961242675781,
            -2656.33056640625
        ],
        [
            -608.64697265625,
            3283.060791015625,
            -1446.4940185546875,
            52.48081970214844,
            -498.0961608886719,
            -1943.2216796875,
            -1253.4010009765625,
            -3117.74462890625,
            -1712.262451171875,
            -2280.43359375
        ]
        ]}
        self.assertEqual(result, expected_value)
        self.assertEqual(response.status_code, 200)

if __name__ == '__main__':
    unittest.main()



import re

image_extensions = ['.jpg', '.jpeg', '.png', '.gif']  # Liste des extensions d'images à vérifier

pattern = r'^(?:[^.]+{})+$'.format('|'.join(map(re.escape, image_extensions)))

df['is_only_image'] = df['dlpfilename'].str.match(pattern, case=False, regex=True).astype(int)
Récapitulation de mes réalisations :

Stage M1 : En tant que stagiaire en Data Science, j'ai intégré l'équipe avec laquelle j'avais déjà établi des liens lors de mon précédent stage. Ayant développé une bonne connaissance des membres de l'équipe Data et du SOC, je me suis senti bien inclus. J'ai principalement travaillé sur des sujets de machine learning, notamment la détection de fuites d'informations via les URL. J'ai activement contribué au développement de fonctionnalités, codant des solutions pour répondre aux besoins de l'équipe.
Stage de fin d'études : J'ai eu l'opportunité de réintégrer l'équipe avec laquelle je me sens à l'aise, travaillant aux côtés de Mathieu et Ismael. Cette familiarité avec les membres de l'équipe Data et du SOC m'a permis de m'intégrer plus facilement et de collaborer de manière efficace. J'ai mis en place MLflow pour l'équipe, en me concentrant sur DLTK. J'ai également développé un premier algorithme pour la partie beacon C2, que j'ai ensuite amélioré. Ensuite, j'ai apporté mon aide à Ismael sur la partie NLP pour la détection de fuites d'informations par e-mail.
Reprendre le cas d'utilisation de Valentin : Ayant déjà une connaissance approfondie de l'équipe et une bonne relation de travail avec les membres de l'équipe Data et du SOC, j'ai pris en charge le cas d'utilisation de Valentin concernant la détection de fuites d'informations via les URL. Étant donné que les critères de classification ont changé, je repars presque de zéro sur ce sujet, avec pour objectif de mettre en production un algorithme efficace dans les plus brefs délais.

Récapitulation de mes réalisations :

Stage M1 : En tant que stagiaire en Data Science, j'ai intégré l'équipe et me suis concentré sur des sujets de machine learning, notamment la détection de fuites d'informations via les URL. J'ai activement contribué au développement de fonctionnalités, codant des solutions pour répondre aux besoins de l'équipe.
Stage de fin d'études : J'ai réintégré l'équipe avec laquelle je me sens à l'aise, travaillant aux côtés de Mathieu et Ismael. J'ai mis en place MLflow pour l'équipe, en me concentrant sur DLTK. J'ai également développé un premier algorithme pour la partie beacon C2, que j'ai ensuite amélioré. Ensuite, j'ai apporté mon aide à Ismael sur la partie NLP pour la détection de fuites d'informations par e-mail.
Reprendre le cas d'utilisation de Valentin : J'ai pris en charge le cas d'utilisation de Valentin concernant la détection de fuites d'informations via les URL. Étant donné que les critères de classification ont changé, je repars presque de zéro sur ce sujet, avec pour objectif de mettre en production un algorithme efficace dans les plus brefs délais.
