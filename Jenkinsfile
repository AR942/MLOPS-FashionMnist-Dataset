pipeline{
    agent any
    
    stages{


        stage('Staging branch'){
            steps{
                bat 'git checkout dev'
                bat 'git pull'
                bat 'git checkout -b staging'
                bat 'git push --set-upstream origin staging'
            }
        }

        stage('Build'){
            steps{
                bat 'pip install -r requirements.txt'
            }
        }

        stage('Test'){
            steps{
                bat 'python -m unittest'
            }
        }

        stage ('Image'){
            steps{
                bat 'docker build -t image .'
                bat 'docker run -d -p 5000:5000 image'
            }
        }

        stage('Merge main and staging'){
            steps{
                bat 'git checkout main'
                bat 'git merge staging'
                bat 'git push --set-upstream origin main'
                bat 'git branch -d staging'
                bat 'git push origin --delete staging'
            }         
  
        }

    }
}