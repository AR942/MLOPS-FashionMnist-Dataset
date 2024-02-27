pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh "pip install -r requirements.txt"
            }
        }
        stage('Test') {
            steps {
                sh "python uni_test.py"
            }
        }
        stage('DockerBuild') {
            steps {
                sh "docker build -t image ."
            }
        }
        stage('DockerRun') {
            steps {
                sh "docker run -d image"
            }
        }
        
    }
}


 Les principaux avantages de cette application sont : 1
 Détection simplifiée des anomalies comportementales : Cette application fournit un flux de travail simplifié pour le déploiement de recherches de métriques comportementales et de règles de notation des anomalies
 Cela réduit la complexité de  mise en œuvre et les procédures de détection des anomalies
 2
 Intégration avec Splunk : En tirant parti des capacités de traitement des données de Splunk, l'application permet une intégration transparente avec les systèmes existants et facilite la collecte et l'analyse des données comportementales
 3
 Analyses avancées : Cette application utilise une architecture à trois niveaux pour transformer les données brutes en profils comportementaux d'entités, permettant une analyse comportementale avancée  et une détection d'anomalies basées sur des critères définis par l'utilisateur
 4
 Tableaux de bord et visualisations : Cette application fournit des tableaux de bord et des visualisations pour surveiller les mesures comportementales de vos entités, vous permettant de mieux comprendre les tendances et les modèles au fil du temps
 5
 Optimisation des performances : l'application fournit une vue permettant de surveiller les performances des règles de notation des anomalies afin que vous puissiez rapidement identifier et  résoudre  les problèmes de performances
 Cependant, avant de décider de lancer cette application, il est important de considérer les fonctionnalités et ressources actuelles, ainsi que les alternatives disponibles
 Nous vous encourageons donc à discuter de ces points plus en détail afin de déterminer la meilleure approche pour notre équipe
 Je serais heureux de discuter de ce sujet plus en détail
 Cordialement, [Votre nom]
