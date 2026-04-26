# 🌿 Iris Classifier — Complete ML Pipeline

A production-ready Machine Learning pipeline that classifies Iris flowers using multiple algorithms, selects the best model via cross-validation, and serves predictions through a REST API deployed with Docker.

---

## 📊 Results Summary

| Model               | CV Accuracy | Test Accuracy | F1-Score |
|---------------------|:-----------:|:-------------:|:--------:|
| Logistic Regression | 0.9583      | 0.9333        | 0.9332   |
| Random Forest       | 0.9500      | 0.9000        | 0.8990   |
| **SVM** ⭐          | **0.9667**  | **0.9667**    | **0.9666** |
| Gradient Boosting   | 0.9667      | 0.9667        | 0.9666   |
| KNN                 | 0.9667      | 0.9333        | 0.9332   |

**Selected Model:** SVM (RBF kernel, C=1, gamma='scale') — 96.67% accuracy

---

## 🏗️ Project Structure

```
ml-pipeline/
├── model/
│   ├── train.py          # Full ML pipeline (EDA → train → evaluate → save)
│   ├── model.pkl         # Trained model (generated at build time)
│   └── metadata.json     # Model metrics, feature ranges, class names
├── app/
│   ├── app.py            # Flask REST API
│   └── templates/
│       └── index.html    # Interactive web UI
├── Dockerfile            # Multi-stage Docker build
├── docker-compose.yml    # Dev + prod compose configs
├── nginx.conf            # Nginx reverse proxy config
├── deploy.sh             # Cloud deployment scripts
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Run with Docker (recommended)

```bash
# Build & run production image
docker build -t iris-classifier .
docker run -p 5000:5000 iris-classifier

# Or use Docker Compose
docker-compose --profile prod up
```

### 2. Run locally (development)

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model
python model/train.py

# Start the API
cd app && python app.py
```

Open **http://localhost:5000** in your browser.

---

## 🔌 API Reference

### Health Check
```http
GET /health
```
```json
{"status": "healthy", "model": "SVM"}
```

### Single Prediction
```http
POST /api/predict
Content-Type: application/json

{
  "sepal length (cm)": 5.1,
  "sepal width (cm)": 3.5,
  "petal length (cm)": 1.4,
  "petal width (cm)": 0.2
}
```
**Response:**
```json
{
  "prediction": "setosa",
  "class_index": 0,
  "confidence": {"setosa": 99.2, "versicolor": 0.5, "virginica": 0.3},
  "input": {"sepal length (cm)": 5.1, ...}
}
```

### Batch Prediction
```http
POST /api/batch_predict
Content-Type: application/json

[
  {"sepal length (cm)": 5.1, "sepal width (cm)": 3.5, "petal length (cm)": 1.4, "petal width (cm)": 0.2},
  {"sepal length (cm)": 6.7, "sepal width (cm)": 3.0, "petal length (cm)": 5.6, "petal width (cm)": 2.1}
]
```

### Model Info & Metrics
```http
GET /api/info
GET /api/metrics
```

---

## ☁️ Cloud Deployment

### Option A: Render.com (Free, Easiest)
1. Push repo to GitHub
2. Connect to [render.com](https://render.com)
3. New → Web Service → Docker → Deploy
4. Set env var: `PORT=5000`

### Option B: Google Cloud Run
```bash
# Authenticate
gcloud auth login

# Deploy
./deploy.sh gcp YOUR_PROJECT_ID us-central1
```

### Option C: AWS EC2
```bash
./deploy.sh aws your_dockerhub_user YOUR_EC2_IP ~/.ssh/key.pem
```

### Option D: Railway
```bash
npm install -g @railway/cli
./deploy.sh railway
```

---

## 🧪 Testing the API

```bash
# Health check
curl http://localhost:5000/health

# Predict setosa
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"sepal length (cm)":5.1,"sepal width (cm)":3.5,"petal length (cm)":1.4,"petal width (cm)":0.2}'

# Get all metrics
curl http://localhost:5000/api/metrics | python3 -m json.tool
```

---

## 🔬 ML Pipeline Details

### Pipeline Steps
1. **Data Loading** — UCI Iris dataset (150 samples, 4 features, 3 classes)
2. **EDA** — Descriptive stats, class distribution, missing value check
3. **Preprocessing** — StandardScaler (zero mean, unit variance)
4. **Model Selection** — 5 classifiers evaluated with 5-fold CV
5. **Hyperparameter Tuning** — GridSearchCV on best model
6. **Evaluation** — Accuracy, F1, confusion matrix, classification report
7. **Serialization** — Pickle + JSON metadata

### Why SVM?
- Highest CV accuracy (0.9667) with lowest variance
- Excellent generalization on test set
- RBF kernel handles non-linear boundaries in feature space
- Probability estimates via Platt scaling

---

## 📦 Docker Architecture

```
┌─────────────────────────────────────┐
│  Multi-stage Dockerfile             │
│                                     │
│  Stage 1 (trainer)                  │
│  ├── Install all deps               │
│  └── Run train.py → model.pkl       │
│                                     │
│  Stage 2 (production)               │
│  ├── Copy model.pkl from stage 1    │
│  ├── Install runtime deps only      │
│  ├── Non-root user (security)       │
│  └── gunicorn WSGI server           │
└─────────────────────────────────────┘
         ↓
┌──────────────────┐    ┌─────────────┐
│  Nginx (port 80) │───→│ Flask :5000 │
└──────────────────┘    └─────────────┘
```

**Benefits:**
- Smaller final image (no training deps)
- Model trained at build time (reproducible)
- Non-root execution (security)
- Health checks for orchestrators (K8s/ECS)

---

## 📄 License
MIT
