**Model‑as‑a‑Service (MaaS) Seed Micro‑service**

`maas‑seed` is a production‑ready template that turns any pickled scikit‑learn model into a fully containerised prediction API. It:

* pulls the **latest model artefact from S3** at startup,
* caches it in **Redis** for hot‑reload‑free inference,
* exposes a **FastAPI** endpoint (`/transaction`) for real‑time scoring, and
* (in dev) tunnels to a **remote Feast Python server** via `kubectl port‑forward`, so you can pull online features without hand‑running `kubectl` every time.

---

## 🗂️ Project Layout

```
maas-seed/
│
├── app/
│   ├── main.py            # FastAPI app + startup hook
│   ├── api.py             # /transaction endpoint + validation
│   ├── model_loader.py    # fetch newest model from S3
│   ├── inference.py       # feature store call → dataframe → predict
│   └── redis_cache.py
│
├── Dockerfile             # FastAPI image (Python 3.10‑slim)
├── docker‑compose.yml     # redis + fastapi + feast‑tunnel
├── requirements.txt
├── maas_txn.sh            # fetch creds, build & run services, clean up
└── README.md              # you are here
```

---

## 🚀 Quick‑start (local dev)

Make the launcher script executable and run it:

```bash
chmod +x maas_txn.sh
./maas_txn.sh
```

This will:

1. Fetch AWS credentials from Secrets Manager and write a temporary `.env`.
2. Build and start all services via Docker Compose.
3. Clean up the `.env` file automatically when done.

---

## 🗂️ Environment Variables

All necessary AWS keys and other settings are handled by `maas_txn.sh`. There’s no need to manage an `aws.env` file manually.

---

## 🏗️ Deploying to Kubernetes / EKS

* Drop the side‑car; deploy `maas‑seed-service` **inside the same cluster** and set `FEAST_BASE_URL=http://feast-python-server.feast-python.svc.cluster.local:6567`.
* Use a **Secret** or IAM role for AWS credentials instead of local files.
* Redis can be external (Elasticache) or an in‑cluster Helm release.

---

## 📄 License

MIT — free to use with attribution.

---

## 👥 Authors

Built by **Natnael** & the Data Science Team. Part of the scalable AI infrastructure at **KFT**.
