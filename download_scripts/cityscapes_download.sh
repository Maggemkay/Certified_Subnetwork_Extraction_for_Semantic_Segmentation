#!/usr/bin/env bash
set -euo pipefail

# Downloads the Cityscapes fine-annotation semantic-segmentation set
# (gtFine_trainvaltest.zip ≈ 241 MB + leftImg8bit_trainvaltest.zip ≈ 11 GB)
# into the project's data/cityscapes folder.
#
# Run once; nothing has to be installed beforehand. The official downloader
# (`csDownload` from the pip package `cityscapesscripts`,
# https://github.com/mcordts/cityscapesScripts) is installed into a TEMPORARY
# virtual environment that is deleted when the script ends, so the active
# Python environment is left untouched.
#
# You need a Cityscapes account (https://www.cityscapes-dataset.com/register/).
# csDownload asks for the username and password, and then asks whether to store
# them unencrypted on disk; answering "n" is fine for a one-off download.
#
# Result after a successful run:
#   data/cityscapes/leftImg8bit/{train,val,test}/<city>/*_leftImg8bit.png
#   data/cityscapes/gtFine/{train,val,test}/<city>/*_gtFine_{labelIds,instanceIds,color}.png, *_gtFine_polygons.json
#   data/cityscapes/README, license.txt

# Resolve the project's data/cityscapes folder relative to this script, so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CITYSCAPES_DIR="$SCRIPT_DIR/../data/cityscapes"

PACKAGE_NAMES=("gtFine_trainvaltest.zip" "leftImg8bit_trainvaltest.zip")

mkdir -p "$CITYSCAPES_DIR"
cd "$CITYSCAPES_DIR"

if [ -d "gtFine" ] || [ -d "leftImg8bit" ]; then
    echo "[cityscapes] ERROR: $CITYSCAPES_DIR/gtFine or $CITYSCAPES_DIR/leftImg8bit already exists. Remove them first to re-download."
    exit 1
fi

# --- Temporary virtual environment holding cityscapesscripts ---------------------------
TEMPORARY_VENV_DIR="$(mktemp -d -t cityscapes_download_venv.XXXXXX)"
remove_temporary_venv() {
    echo "[cityscapes] Removing the temporary virtual environment $TEMPORARY_VENV_DIR"
    rm -rf "$TEMPORARY_VENV_DIR"
}
trap remove_temporary_venv EXIT

echo "[cityscapes] Creating a temporary virtual environment in $TEMPORARY_VENV_DIR (python: $(command -v python3))"
python3 -m venv "$TEMPORARY_VENV_DIR"
echo "[cityscapes] Installing cityscapesscripts into the temporary virtual environment ..."
"$TEMPORARY_VENV_DIR/bin/python" -m pip install --quiet --upgrade pip
"$TEMPORARY_VENV_DIR/bin/python" -m pip install --quiet cityscapesscripts
CSDOWNLOAD="$TEMPORARY_VENV_DIR/bin/csDownload"

# --- Download -----------------------------------------------------------------------------
# Zips that are already complete (passed the integrity test in an earlier, interrupted
# run) are not downloaded again.
for PACKAGE_NAME in "${PACKAGE_NAMES[@]}"; do
    if [ -f "$PACKAGE_NAME" ] && unzip -tq "$PACKAGE_NAME" > /dev/null 2>&1; then
        echo "[cityscapes] $PACKAGE_NAME is already present and passes the zip integrity test; skipping its download."
        continue
    fi
    rm -f "$PACKAGE_NAME"
    echo "[cityscapes] Downloading $PACKAGE_NAME into $CITYSCAPES_DIR (you will be asked for your Cityscapes credentials) ..."
    "$CSDOWNLOAD" -d "$CITYSCAPES_DIR" "$PACKAGE_NAME"
done

# --- Integrity test -----------------------------------------------------------------------
for PACKAGE_NAME in "${PACKAGE_NAMES[@]}"; do
    echo "[cityscapes] $PACKAGE_NAME size: $(stat -c %s "$PACKAGE_NAME") bytes"
    echo "[cityscapes] Testing zip integrity of $PACKAGE_NAME (unzip -t) ..."
    if unzip -tq "$PACKAGE_NAME" > /dev/null; then
        echo "[cityscapes] Zip integrity test PASSED for $PACKAGE_NAME."
    else
        echo "[cityscapes] ERROR: zip integrity test FAILED for $PACKAGE_NAME. The zip is kept at $CITYSCAPES_DIR/$PACKAGE_NAME."
        echo "[cityscapes] Delete it and run this script again to re-download."
        exit 1
    fi
done

# --- Extract ------------------------------------------------------------------------------
# Both zips contain their own README and license.txt; -o overwrites them without prompting.
for PACKAGE_NAME in "${PACKAGE_NAMES[@]}"; do
    echo "[cityscapes] Extracting $PACKAGE_NAME ..."
    unzip -q -o "$PACKAGE_NAME"
    rm "$PACKAGE_NAME"
done

echo "[cityscapes] Done. Image counts:"
for SPLIT_NAME_AND_EXPECTED_COUNT in "train 2975" "val 500" "test 1525"; do
    read -r SPLIT_NAME EXPECTED_COUNT <<< "$SPLIT_NAME_AND_EXPECTED_COUNT"
    echo "[cityscapes]   $SPLIT_NAME images (leftImg8bit):  $(find "leftImg8bit/$SPLIT_NAME" -name '*_leftImg8bit.png' | wc -l) (expected $EXPECTED_COUNT)"
    echo "[cityscapes]   $SPLIT_NAME labels (gtFine labelIds): $(find "gtFine/$SPLIT_NAME" -name '*_gtFine_labelIds.png' | wc -l) (expected $EXPECTED_COUNT)"
done
echo "[cityscapes] Note: the test-split labels are placeholders (the real ones are held back by the benchmark server)."
