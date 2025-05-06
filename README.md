# maas-seed

**Model-as-a-Service (MaaS) Seed Microservice**

`maas-seed` is a production-ready microservice template that exposes machine learning models as APIs. It loads the latest `.pkl` model from a directory, caches it in Redis, and provides a FastAPI-based HTTP endpoint for real-time predictions. Designed for scalable deployment on Docker and Kubernetes (e.g., EKS).

---

## 🚀 Purpose

- Serve ML models through a REST API
- Load the latest `.pkl` model automatically at startup
- Cache models in memory using Redis for fast inference
- Enable reattempts and async processing for scalability
- Containerized for cloud-native deployment

---

## 🗂️ Project Structure

maas-seed/
│
├── app/
│ ├── main.py # FastAPI app and lifecycle
│ ├── model_loader.py # Loads latest model from file
│ └── redis_cache.py # Caches model in Redis
│
├── models/ # Place your .pkl model files here
├── Dockerfile
├── requirements.txt
└── README.md
---

## 🐳 Docker Instructions

### 🔨 Build the Docker Image

```bash
docker build -t model-service .
🚀 Run the Docker Container
bash
Copy code
docker run -d \
  -p 8003:8000 \
  -v $(pwd)/models:/models \
  -e MODEL_DIR=/models \
  -e REDIS_URL=redis://host.docker.internal:6379/0 \
  --name model-service \
  model-service
📝 Note (macOS/Windows): host.docker.internal allows Docker to access services running on your host machine (e.g., Redis).
🐧 Note (Linux): Replace host.docker.internal with your host IP (e.g., 172.17.0.1). This must be done for our EKS deployment and aws based test as both are linux based.

📬 API Endpoint
POST /transaction -- chage this to model level names
Request:

json
{
  "APIrequest_data": {
    "source_bank": "string",
    "customerId": "string",
    "loan_type": "string",
    "loan_subtype": "string"
  }

Response:

json
{
  "prediction": "output_value"
}

🧪 Local Testing
Ensure Redis is running locally:

bash

redis-cli ping
# Output: PONG
Use curl to test the endpoint:

bash

curl -X POST http://localhost:8003/predict \
  -H "Content-Type: application/json" \
  -d '{"APIrequest_data": {"source_bank": "X", "customerId": "123", "loan_type": "Y", "loan_subtype": "Z"}}'
✅ Requirements (For Local Dev)
bash

pip install -r requirements.txt
📦 Environment Variables
Variable	Description	Example
MODEL_DIR	Directory where model .pkl files are stored	/models
REDIS_URL	Redis server connection URL	redis://localhost:6379/0

📄 License
MIT License. Free to use with attribution.

👤 Author
Built by Natnael and Data Science Team
Part of the scalable AI infrastructure at KFT.





