#!/bin/bash
# =========================================================
#  ML Pipeline - Cloud Deployment Scripts
#  Supports: AWS EC2, Google Cloud Run, Render.com
# =========================================================

set -euo pipefail

IMAGE_NAME="iris-classifier"
IMAGE_TAG="latest"
FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"

# ─────────────────────────────────────────────────────────
# 0. Build & Test Locally
# ─────────────────────────────────────────────────────────
build_local() {
  echo "🔨 Building Docker image..."
  docker build -t ${FULL_IMAGE} .
  echo "✅ Build complete"
}

test_local() {
  echo "🧪 Testing locally..."
  docker run -d --name iris-test -p 5000:5000 ${FULL_IMAGE}
  sleep 5
  curl -sf http://localhost:5000/health && echo " — Health OK"
  curl -s -X POST http://localhost:5000/api/predict \
    -H "Content-Type: application/json" \
    -d '{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}' \
    | python3 -m json.tool
  docker stop iris-test && docker rm iris-test
  echo "✅ Tests passed"
}

# ─────────────────────────────────────────────────────────
# 1. Deploy to AWS EC2 via Docker Hub
# ─────────────────────────────────────────────────────────
deploy_aws() {
  DOCKER_HUB_USER="${1:-your_dockerhub_user}"
  EC2_IP="${2:-your-ec2-public-ip}"
  EC2_KEY="${3:-~/.ssh/ec2-key.pem}"

  echo "🚀 Deploying to AWS EC2..."
  
  # Push to Docker Hub
  docker tag ${FULL_IMAGE} ${DOCKER_HUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}
  docker push ${DOCKER_HUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}

  # SSH into EC2 and deploy
  ssh -i ${EC2_KEY} ec2-user@${EC2_IP} <<EOF
    sudo yum update -y && sudo yum install -y docker
    sudo service docker start
    sudo docker pull ${DOCKER_HUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}
    sudo docker stop iris-prod 2>/dev/null || true
    sudo docker rm iris-prod   2>/dev/null || true
    sudo docker run -d --name iris-prod \
      -p 80:5000 \
      --restart unless-stopped \
      ${DOCKER_HUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}
    echo "✅ EC2 deployment complete"
EOF
  echo "🌐 App running at: http://${EC2_IP}"
}

# ─────────────────────────────────────────────────────────
# 2. Deploy to Google Cloud Run
# ─────────────────────────────────────────────────────────
deploy_gcp() {
  PROJECT_ID="${1:-your-gcp-project}"
  REGION="${2:-us-central1}"
  
  echo "🚀 Deploying to Google Cloud Run..."

  # Authenticate and configure
  gcloud config set project ${PROJECT_ID}
  
  # Build with Cloud Build
  gcloud builds submit --tag gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG}
  
  # Deploy to Cloud Run
  gcloud run deploy iris-classifier \
    --image gcr.io/${PROJECT_ID}/${IMAGE_NAME}:${IMAGE_TAG} \
    --platform managed \
    --region ${REGION} \
    --port 5000 \
    --memory 512Mi \
    --cpu 1 \
    --min-instances 0 \
    --max-instances 10 \
    --allow-unauthenticated \
    --set-env-vars PORT=5000
  
  echo "✅ Cloud Run deployment complete"
  gcloud run services describe iris-classifier --region ${REGION} --format="value(status.url)"
}

# ─────────────────────────────────────────────────────────
# 3. Deploy to Render.com (render.yaml based)
# ─────────────────────────────────────────────────────────
setup_render() {
  cat > render.yaml <<'YAML'
services:
  - type: web
    name: iris-classifier
    env: docker
    dockerfilePath: ./Dockerfile
    plan: free
    envVars:
      - key: PORT
        value: 5000
    healthCheckPath: /health
YAML
  echo "✅ render.yaml created. Push to GitHub and connect at https://render.com"
}

# ─────────────────────────────────────────────────────────
# 4. Deploy to Railway
# ─────────────────────────────────────────────────────────
deploy_railway() {
  echo "🚀 Deploying to Railway..."
  # Install Railway CLI if needed: npm install -g @railway/cli
  railway login
  railway up --service iris-classifier
  echo "✅ Railway deployment complete"
}

# ─────────────────────────────────────────────────────────
# Main dispatcher
# ─────────────────────────────────────────────────────────
case "${1:-help}" in
  build)     build_local ;;
  test)      test_local ;;
  aws)       deploy_aws "$2" "$3" "$4" ;;
  gcp)       deploy_gcp "$2" "$3" ;;
  render)    setup_render ;;
  railway)   deploy_railway ;;
  *)
    echo "Usage: $0 {build|test|aws <user> <ec2-ip> <key>|gcp <project> <region>|render|railway}"
    ;;
esac
