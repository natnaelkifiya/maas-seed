#!/bin/bash
# Retrieve secret from AWS Secrets Manager
SECRET_JSON=$(aws secretsmanager get-secret-value --secret-id maas/maas-seed/s3-cred --query SecretString --output text)
export AWS_ACCESS_KEY_ID=$(echo $SECRET_JSON | jq -r .AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(echo $SECRET_JSON | jq -r .AWS_SECRET_ACCESS_KEY)
# Start Docker Compose
docker-compose up -d --build
