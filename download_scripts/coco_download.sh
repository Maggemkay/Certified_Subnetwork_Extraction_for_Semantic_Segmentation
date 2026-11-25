#!/usr/bin/env bash
set -euo pipefail

# Resolve the project's data/coco folder relative to this script, so it works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COCO_DIR="$SCRIPT_DIR/../data/coco"

mkdir -p "$COCO_DIR/images"
cd "$COCO_DIR/images"

wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/zips/test2017.zip

unzip train2017.zip
unzip val2017.zip
unzip test2017.zip

rm train2017.zip
rm val2017.zip
rm test2017.zip
cd ../
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/image_info_test2017.zip

unzip annotations_trainval2017.zip
unzip stuff_annotations_trainval2017.zip
unzip image_info_test2017.zip

rm annotations_trainval2017.zip
rm stuff_annotations_trainval2017.zip
rm image_info_test2017.zip
