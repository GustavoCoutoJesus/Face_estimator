#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo "============================================================================"
echo "  Social Judgment Face Estimator (Todorov & Peterson OMI - 34 Attributes)"
echo "============================================================================"

mkdir -p input_faces

if ! python3 -c "import face_trait_transformer, mediapipe, cv2, torch" >/dev/null 2>&1; then
    echo "[1/2] Installing required Python packages from requirements.txt..."
    python3 -m pip install -r requirements.txt
fi

echo "Running face estimation on images in ./input_faces ..."
python3 estimate_faces.py input_faces --output-dir results
