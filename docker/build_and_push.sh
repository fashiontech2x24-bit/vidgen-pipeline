#!/usr/bin/env bash
# Build the server image and push to a registry.
#   DOCKER_USER=youruser IMAGE=vace-vid-server TAG=v1 bash docker/build_and_push.sh
set -euo pipefail

DOCKER_USER="${DOCKER_USER:?set DOCKER_USER (your Docker Hub username)}"
IMAGE="${IMAGE:-vace-vid-server}"
TAG="${TAG:-latest}"
REPO="${DOCKER_USER}/${IMAGE}:${TAG}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo ">> Building ${REPO} (platform linux/amd64)"
docker build --platform linux/amd64 -f docker/Dockerfile -t "${REPO}" .

if [[ "${PUSH:-1}" == "1" ]]; then
  echo ">> Pushing ${REPO}"
  docker push "${REPO}"
  echo ">> Done: ${REPO}"
fi
