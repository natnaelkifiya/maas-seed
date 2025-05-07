
# maas‑seed

**Model‑as‑a‑Service (MaaS) Seed Micro‑service**

`maas‑seed` is a production‑ready template that turns any pickled
scikit‑learn model into a fully containerised prediction API.  It

* pulls the **latest model artefact from S3** at startup,  
* caches it in **Redis** for hot‑reload‑free inference,  
* exposes a **FastAPI** endpoint (`/transaction`) for real‑time scoring, and  
* (in dev) tunnels to a **remote Feast Python server** via `kubectl
  port‑forward`, so you can pull online features without hand‑running
  `kubectl` every time.

---

## 🗂️ Project Layout

```

maas-seed/
│
├── app/
│   ├── main.py            # FastAPI app + startup hook
│   ├── api.py             # /transaction endpoint + validation
│   ├── model\_loader.py    # fetch newest model from S3
│   ├── inference.py       # feature store call → dataframe → predict
│   └── redis\_cache.py
│
├── Dockerfile             # FastAPI image (Python 3.10‑slim)
├── docker‑compose.yml     # redis + fastapi + feast‑tunnel
├── requirements.txt
├── aws.env                # AWS creds, Redis & MLflow URIs
└── README.md              # you are here

````

---

## 🚀 Quick‑start (local dev)

> **Prereqs:** Docker ≥ 20.10 with Compose v2, an AWS user with
> `eks:DescribeCluster` + S3 read permissions, and a running Kubernetes
> cluster that hosts the **Feast Python Server** service
> `feast-python-server` in namespace `feast-python`.

### 1 Clone and create `aws.env`

```env
# aws.env  (don't commit with git)
AWS_ACCESS_KEY_ID=…
AWS_SECRET_ACCESS_KEY=…
MLFLOW_TRACKING_URI=http://3.239.37.210:5000
REDIS_URL=redis://redis:6379/0                  # resolves inside compose
````

### 2 Start everything

```bash
docker compose up -d --build        # builds FastAPI image, then spins up
```

Compose brings up **three** containers:

| Container           | Purpose                                                                                                      |
| ------------------- | ------------------------------------------------------------------------------------------------------------ |
| `redis`             | low‑latency model cache                                                                                      |
| `maas-seed-service` | FastAPI app (port `8000`)                                                                                    |
| `feast-tunnel`      | `aws eks update-kubeconfig` → `kubectl port‑forward` tunneled to `localhost:6567` inside the compose network |

Watch logs:

```bash
docker compose logs -f feast-tunnel       # expect “Forwarding from 127.0.0.1:6567”
docker compose logs -f maas-seed-service  # expect “Application startup complete.”
```

### 3 Call the API

```bash
curl -X POST http://localhost:8000/transaction \
  -H "Content-Type: application/json" \
  -d '{
        "customerId":  "cust_test",
        "loan_type":   "msme",
        "loan_subtype":"demo",
        "source_bank": "coop"
      }' | jq .
```

Example response

```json
{
  "credit_score": "418",
  "feature_importance": [
    {"total_credit": 0.33},
    {"log_balance": 0.20},
    {"total_debet": 0.14},
    {"volatility": 0.11},
    {"sd_credit": 0.09}
  ],
  "customerId": "cust_test"
}
```

Interactive docs live at **[http://localhost:8000/docs](http://localhost:8000/docs)**.

### 4 Tear down

```bash
docker compose down
```

---

## 🧩 Environment Variables

| Variable                                     | Description                                         | Default in compose                 |
| -------------------------------------------- | --------------------------------------------------- | ---------------------------------- |
| `REDIS_URL`                                  | Redis connection string                             | `redis://redis:6379/0`             |
| `FEAST_BASE_URL`                             | URL the API uses to reach Feast                     | `http://localhost:6567` (tunneled) |
| `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` | Used by side‑car to run `aws eks update-kubeconfig` | from `aws.env`                     |
| `MLFLOW_TRACKING_URI`                        | Optional MLflow tracking server                     | from `aws.env`                     |

---

## 🏗️ Deploying to Kubernetes / EKS

* Drop the side‑car; deploy `maas‑seed-service` **inside the same cluster** and
  set `FEAST_BASE_URL=http://feast-python-server.feast-python.svc.cluster.local:6567`.
* Use a **Secret** or IAM role for AWS credentials instead of `aws.env`.
* Redis can be external (Elasticache) or an in‑cluster Helm release.

---

## 📄 License

MIT — free to use with attribution.

---

## 👥 Authors

Built by **Natnael** & the Data Science Team. Part of the scalable AI
infrastructure at **KFT**.

```
```
