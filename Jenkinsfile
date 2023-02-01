pipeline {
    agent any

    stages {
        stage('Build') {
            steps {
                sh "pip3 install -r requirements.txt"
            }
        }
        stage('Test') {
            steps {
                sh "python3 test_main.py"
            }
        }
        stage('DockerBuild') {
            steps {
                sh "docker build -t rfmlep ."
            }
        }
        stage('DockerRun') {
            steps {
                sh "docker run -d rfmlep"
            }
        }
        
    }
}