#!/usr/bin/env bash
set -euo pipefail

# Downloads the ADE20K scene-parsing benchmark (ADEChallengeData2016, 150 classes)
# into the project's data/ade20k folder.
#
# Result after a successful run:
#   data/ade20k/ADEChallengeData2016/images/{training,validation}/*.jpg
#   data/ade20k/ADEChallengeData2016/annotations/{training,validation}/*.png
#   data/ade20k/ADEChallengeData2016/objectInfo150.txt, sceneCategories.txt

# Resolve the project's data/ade20k folder relative to this script, so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ADE20K_DIR="$SCRIPT_DIR/../data/ade20k"

ZIP_URL="https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
ZIP_NAME="ADEChallengeData2016.zip"
# Expected size: the server's Content-Length for the file dated 2022-07-22.
# A mismatch only prints a warning; `unzip -t` below is the hard integrity check.
EXPECTED_ZIP_SIZE_IN_BYTES=967382037

mkdir -p "$ADE20K_DIR"
cd "$ADE20K_DIR"

if [ -d "ADEChallengeData2016" ]; then
    echo "[ade20k] ERROR: $ADE20K_DIR/ADEChallengeData2016 already exists. Remove it first to re-download."
    exit 1
fi

echo "[ade20k] Downloading $ZIP_URL into $ADE20K_DIR"
# -c resumes a partial download if the script was interrupted earlier.
wget -c -O "$ZIP_NAME" "$ZIP_URL"

ACTUAL_ZIP_SIZE_IN_BYTES=$(stat -c %s "$ZIP_NAME")
echo "[ade20k] Downloaded zip size: $ACTUAL_ZIP_SIZE_IN_BYTES bytes (expected $EXPECTED_ZIP_SIZE_IN_BYTES bytes)"
if [ "$ACTUAL_ZIP_SIZE_IN_BYTES" -ne "$EXPECTED_ZIP_SIZE_IN_BYTES" ]; then
    echo "[ade20k] WARNING: zip size differs from the expected size. Continuing to the zip integrity test."
fi

echo "[ade20k] Testing zip integrity (unzip -t) ..."
if unzip -tq "$ZIP_NAME" > /dev/null; then
    echo "[ade20k] Zip integrity test PASSED."
else
    echo "[ade20k] ERROR: zip integrity test FAILED. The zip is kept at $ADE20K_DIR/$ZIP_NAME."
    echo "[ade20k] Delete it and run this script again to re-download."
    exit 1
fi

echo "[ade20k] Extracting $ZIP_NAME ..."
unzip -q "$ZIP_NAME"

rm "$ZIP_NAME"

echo "[ade20k] Done. Image counts:"
echo "[ade20k]   training images:        $(find ADEChallengeData2016/images/training -name '*.jpg' | wc -l) (expected 20210)"
echo "[ade20k]   validation images:      $(find ADEChallengeData2016/images/validation -name '*.jpg' | wc -l) (expected 2000)"
echo "[ade20k]   training annotations:   $(find ADEChallengeData2016/annotations/training -name '*.png' | wc -l) (expected 20210)"
echo "[ade20k]   validation annotations: $(find ADEChallengeData2016/annotations/validation -name '*.png' | wc -l) (expected 2000)"
