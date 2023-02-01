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