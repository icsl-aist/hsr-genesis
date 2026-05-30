#!/usr/bin/env bash
set -euo pipefail

# ────────────────────────────────────────────────────────
# hsr-genesis Docker runner
#
# Builds (if needed) and runs the Docker development
# environment with GPU, Vulkan, and optional X11 forwarding.
#
# Usage:
#   ./scripts/docker-run.sh                          # run tests
#   ./scripts/docker-run.sh --viewer                 # run tests with viewer
#   ./scripts/docker-run.sh -- python my_script.py   # run a custom script
#   ./scripts/docker-run.sh --viewer -- python my_script.py
#   ./scripts/docker-run.sh -- bash                  # interactive shell
# ────────────────────────────────────────────────────────

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
IMAGE_NAME="hsr-genesis"
DOCKERFILE="$REPO_ROOT/Dockerfile"

HAS_NVIDIA_GPU="false"
if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
    HAS_NVIDIA_GPU="true"
fi

USE_VIEWER="false"
USER_ARGS=()
for arg in "$@"; do
    case "$arg" in
        --viewer|--view)
            USE_VIEWER="true"
            ;;
        --)
            # everything after "--" is the command to run inside the container
            shift
            USER_ARGS=("$@")
            break
            ;;
        *)
            USER_ARGS+=("$arg")
            ;;
    esac
done

# Determine the command to run inside the container
if [ ${#USER_ARGS[@]} -gt 0 ]; then
    CMD=("${USER_ARGS[@]}")
else
    # Default: run tests
    CMD=(bash -c "PYTHONPATH=src python -m pytest tests/ -x -o addopts= 2>&1")
fi

echo "──────────────────────────────────────────────"
echo " hsr-genesis Docker runner"
echo "──────────────────────────────────────────────"
echo " GPU available : $HAS_NVIDIA_GPU"
echo " Viewer enabled: $USE_VIEWER"
if [ ${#USER_ARGS[@]} -gt 0 ]; then
    echo " Command       : ${USER_ARGS[*]}"
else
    echo " Command       : pytest (all tests)"
fi
echo "──────────────────────────────────────────────"

# Build the image if needed
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "==> Building Docker image '$IMAGE_NAME'..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" "$REPO_ROOT"
    echo "==> Build complete."
fi

# Common Docker arguments
DOCKER_ARGS=(
    --rm
    -v "$REPO_ROOT:/workspace"
    -w /workspace
    -e PYTHONPATH=src
    -e NVIDIA_DRIVER_CAPABILITIES=all
)

if [ "$HAS_NVIDIA_GPU" = "true" ]; then
    DOCKER_ARGS+=(--gpus all)
fi

if [ "$USE_VIEWER" = "true" ]; then
    DOCKER_ARGS+=(
        -e DISPLAY="$DISPLAY"
        -v /tmp/.X11-unix:/tmp/.X11-unix:ro
        --network host
    )
    echo "==> (X11 forwarded — make sure 'xhost +local:docker' was run)"
fi

exec docker run "${DOCKER_ARGS[@]}" "$IMAGE_NAME" "${CMD[@]}"
