#!/usr/bin/env bash
set -euo pipefail

###############################################################################
# 0)  Sanity-check: aws CLI must exist
###############################################################################
command -v aws >/dev/null || {
  echo "aws CLI not found. Install AWS CLI v2 first:"
  echo "  https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
}

###############################################################################
# 1)  Ensure jq exists (install silently if missing)
###############################################################################
if ! command -v jq >/dev/null 2>&1; then
  echo "🔍 jq not found — installing…"

  if   command -v apt-get >/dev/null 2>&1; then sudo apt-get update -qq && sudo apt-get install -y -qq jq
  elif command -v yum    >/dev/null 2>&1; then sudo yum install -y -q epel-release jq
  elif command -v dnf    >/dev/null 2>&1; then sudo dnf install -y -q jq
  elif command -v apk    >/dev/null 2>&1; then sudo apk add --no-cache jq
  elif command -v brew   >/dev/null 2>&1; then brew install jq
  else
    echo "Unsupported package manager. Install jq manually: https://jqlang.org/download/"
    exit 1
  fi
fi

###############################################################################
# 2)  Pull AWS creds from Secrets Manager
###############################################################################
SECRET_JSON=$(aws secretsmanager get-secret-value \
  --secret-id maas/maas-seed/s3-cred \
  --query SecretString --output text)

export AWS_ACCESS_KEY_ID=$(echo "$SECRET_JSON" | jq -r .AWS_ACCESS_KEY_ID)
export AWS_SECRET_ACCESS_KEY=$(echo "$SECRET_JSON" | jq -r .AWS_SECRET_ACCESS_KEY)

###############################################################################
# 3)  Docker-Compose namespace (avoids name clashes)
###############################################################################
PROJECT="maas"   # <-- change if you run multiple stacks side-by-side

###############################################################################
# 4)  Bring down any previous stack for this project only
###############################################################################
echo "➡  Stopping old '${PROJECT}' stack (if any)…"
docker compose -p "$PROJECT" down --remove-orphans --volumes

###############################################################################
# 5)  OPTIONAL: purge stray global containers with hard-coded names
#     (uncomment list if your compose file uses 'container_name: redis', etc.)
###############################################################################
for stale in redis postgres; do
  if docker ps -a --format '{{.Names}}' | grep -wq "$stale"; then
    echo "Removing stale container '$stale'"
    docker rm -f "$stale"
  fi
done

###############################################################################
# 6)  Build & launch fresh
###############################################################################
echo "Building and starting '${PROJECT}' stack…"
docker compose -p "$PROJECT" up -d --build

echo "'${PROJECT}' stack is up."
