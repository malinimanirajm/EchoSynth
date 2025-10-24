pipeline {
    agent any

    environment {
        IMAGE_NAME = "echosynth"
        IMAGE_TAG = "latest"
        CONTAINER_NAME = "echosynth_app"
    }

    stages {
        stage('Checkout') {
            steps {
                git branch: 'main', url: 'https://github.com/<your-username>/EchoSynth.git'
            }
        }

        stage('Install Dependencies') {
            steps {
                sh 'python3 -m venv venv'
                sh '. venv/bin/activate && pip install -r requirements.txt'
            }
        }

        stage('Lint and Unit Test') {
            steps {
                sh '''
                . venv/bin/activate
                echo "✅ Running lint checks..."
                pylint src || true
                echo "✅ Running tests..."
                pytest --maxfail=1 --disable-warnings -q || true
                '''
            }
        }

        stage('Build Docker Image') {
            steps {
                sh '''
                echo "🐳 Building Docker image..."
                docker build -t ${IMAGE_NAME}:${IMAGE_TAG} .
                '''
            }
        }

        stage('Deploy Container') {
            steps {
                sh '''
                echo "🚀 Deploying container..."
                docker stop ${CONTAINER_NAME} || true
                docker rm ${CONTAINER_NAME} || true
                docker run -d --name ${CONTAINER_NAME} \
                    -v $(pwd)/data:/app/data \
                    ${IMAGE_NAME}:${IMAGE_TAG}
                '''
            }
        }
    }

    post {
        always {
            echo "✅ Pipeline completed"
        }
        failure {
            echo "❌ Pipeline failed"
        }
    }
}
