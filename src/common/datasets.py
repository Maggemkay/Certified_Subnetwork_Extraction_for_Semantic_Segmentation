"""Dataset adapters. Every adapter has the same interface, so later stages do not
depend on which dataset is used:

    ds.ids[i]    image id (file stem)
    ds.image(i)  PIL RGB image at native resolution
    ds.mask(i)   uint8 array [H, W] of VOC class indices, 255 = void
    ds[i]        (image, mask)

Datasets are selected by name (config: DISCOVERY_DATASET, EVAL_DATASETS, LABEL_REMAPPED_EVAL_DATASETS):
    "voc_sbd_train"                    VOC+SBD train, 10,582 images (VOC val excluded)
    "voc_val"                          VOC 2012 val
    "coco_train2017_voc"               COCO train2017 (118,287 images, the model's training split), same masks
    "coco_val2017_voc"                 COCO val2017 (5,000 images), masks of the 20 VOC categories
    "ade20k_val_voc"                   ADE20K SceneParsing validation (2,000 images), 150 classes remapped to VOC
    "cityscapes_val_voc"               Cityscapes val, gtFine (500 images), labelIds remapped to VOC

Label-remapped datasets (LABEL_REMAPPED_DATASETS: COCO, ADE20K, Cityscapes) also have
    ds.source_mask(i)  uint8 array [H, W] of the source dataset's own label values (before the remapping)
    ds.source_classes  [(source value, source name, VOC class or 255)], where each source value goes
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import SBDataset, VOCSegmentation

import config as C

VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
    "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
    "train", "tvmonitor",
]
NUM_CLASSES = len(VOC_CLASSES)  # 21
VOID_LABEL = 255


def voc_palette():
    """[256, 3] uint8 colours of the VOC palette (void = 255 is light beige)."""
    palette = np.zeros((256, 3), dtype=np.uint8)
    for i in range(256):
        c, r, g, b = i, 0, 0, 0
        for j in range(8):
            r |= ((c >> 0) & 1) << (7 - j)
            g |= ((c >> 1) & 1) << (7 - j)
            b |= ((c >> 2) & 1) << (7 - j)
            c >>= 3
        palette[i] = (r, g, b)
    return palette


VOC_DIR = C.DATA_DIR / "voc"
SBD_DIR = C.DATA_DIR / "sbd"
COCO_DIR = C.DATA_DIR / "coco"
ADE20K_DIR = C.DATA_DIR / "ade20k" / "ADEChallengeData2016"
CITYSCAPES_DIR = C.DATA_DIR / "cityscapes"

# COCO category id of each VOC class (index = VOC class; torchvision's CAT_LIST)
_CAT_LIST = [0, 5, 2, 16, 9, 44, 6, 3, 17, 62, 21, 67, 18, 19, 4, 1, 64, 20, 63, 7, 72]
COCO_TO_VOC = {coco_id: voc_class for voc_class, coco_id in enumerate(_CAT_LIST) if voc_class > 0}

# ------------------------------------------------------------------
# Label remapping into the VOC label space. Rules:
#   - a source class that clearly is a VOC class -> that class (several source classes may merge into one);
#   - a source class that is clearly none of the 20 VOC classes -> 0 (background);
#   - a source class that may contain VOC-class pixels but cannot be mapped one-to-one -> 255 (ignore);
#   - the source dataset's own void / unlabelled pixels -> 255.
# Each mapping is a 256-entry lookup table on the label PNG; a value that is not in the class list maps to 255.
# ------------------------------------------------------------------
# ADE20K-150 (objectInfo150.txt): (Idx, the EXACT full Name string, VOC class or VOID_LABEL). Matched on the whole name,
# never on a substring: synonyms are part of the name ("pot, flowerpot"; "toilet, can, commode, crapper, pot, potty,
# stool, throne"), so a substring match on chair, table, pot, screen or stool would pick the wrong classes. Every class
# not listed here -> 0 (background); 0 ("other / unlabelled") -> 255.
ADE20K_TO_VOC = [
    (91, "airplane, aeroplane, plane", 1),
    (128, "bicycle, bike, wheel, cycle", 2),
    (77, "boat", 4), (104, "ship", 4),
    (99, "bottle", 5),
    (81, "bus, autobus, coach, charabanc, double-decker, jitney, motorbus, motorcoach, omnibus, passenger vehicle", 6),
    (21, "car, auto, automobile, machine, motorcar", 7),
    (103, "van", VOID_LABEL), (84, "truck, motortruck", VOID_LABEL),
    (20, "chair", 9), (31, "armchair", 9), (76, "swivel chair", 9),
    (32, "seat", VOID_LABEL), (111, "stool", VOID_LABEL), (70, "bench", VOID_LABEL),
    (16, "table", 11),
    (34, "desk", VOID_LABEL), (65, "coffee table, cocktail table", VOID_LABEL), (74, "kitchen island", VOID_LABEL),
    (57, "pool table, billiard table, snooker table", VOID_LABEL),
    (117, "minibike, motorbike", 14),
    (13, "person, individual, someone, somebody, mortal, soul", 15),
    # VOC's pottedplant covers plants, pots and vases, so pot, vase and flower -> pottedplant. ADE "plant" is any
    # plant, incl. shrubs and hedges (trees, palms and grass are separate classes), and does not translate well to
    # pottedplant -> 255.
    (126, "pot, flowerpot", 16), (136, "vase", 16), (67, "flower", 16),
    (18, "plant, flora, plant life", VOID_LABEL),
    (24, "sofa, couch, lounge", 18),
    (90, "television receiver, television, television set, tv, tv set, idiot box, boob tube, telly, goggle box", 20),
    (144, "monitor, monitoring device", 20), (142, "crt screen", 20),
    (75, "computer, computing machine, computing device, data processor, electronic computer, information processing "
         "system", VOID_LABEL),
    (131, "screen, silver screen, projection screen", VOID_LABEL),
    (127, "animal, animate being, beast, brute, creature, fauna", VOID_LABEL),  # bird, cat, cow, dog, horse, sheep
]

# Cityscapes labelIds (cityscapesscripts/helpers/labels.py). License plate (-1) never occurs in a *_labelIds.png: the
# label tool skips negative ids, so those pixels keep the car label underneath.
CITYSCAPES_LABEL_NAMES = {
    0: "unlabeled", 1: "ego vehicle", 2: "rectification border", 3: "out of roi", 4: "static", 5: "dynamic",
    6: "ground", 7: "road", 8: "sidewalk", 9: "parking", 10: "rail track", 11: "building", 12: "wall", 13: "fence",
    14: "guard rail", 15: "bridge", 16: "tunnel", 17: "pole", 18: "polegroup", 19: "traffic light",
    20: "traffic sign", 21: "vegetation", 22: "terrain", 23: "sky", 24: "person", 25: "rider", 26: "car",
    27: "truck", 28: "bus", 29: "caravan", 30: "trailer", 31: "train", 32: "motorcycle", 33: "bicycle",
}
CITYSCAPES_TO_VOC = {  # every labelId not listed -> 0 (background)
    24: 15, 25: 15,              # person, rider -> person (VOC labels riders as person)
    26: 7, 28: 6, 31: 19,        # car, bus, train (includes trams)
    32: 14, 33: 2,               # motorcycle -> motorbike, bicycle
    27: VOID_LABEL, 29: VOID_LABEL, 30: VOID_LABEL,   # truck, caravan, trailer: ambiguous
    5: VOID_LABEL,               # dynamic: movable objects of no other class
    4: VOID_LABEL,               # static: void in Cityscapes' own benchmark; fixed
                                 # objects of no other class, may hold chair- or plant-like objects
    0: VOID_LABEL, 1: VOID_LABEL, 2: VOID_LABEL, 3: VOID_LABEL,   # unlabeled, ego vehicle, rectification border, out of roi
}
CITYSCAPES_SOURCE_CLASSES = [(value, name, CITYSCAPES_TO_VOC.get(value, 0))
                             for value, name in CITYSCAPES_LABEL_NAMES.items()]

# Per label-remapped dataset: what the source labels are and which VOC classes it can score (evaluable = the classes
# 1-20 some source class maps to). They follow from the mapping, so they are not a per-run choice.
LABEL_REMAPPED_DATASETS = {
    "coco_val2017_voc": {
        "source": "COCO val2017, 80 categories (torchvision's CAT_LIST: the other 60 -> background, overlapping VOC "
                  "instances -> 255)",
        "evaluable_classes": list(range(1, NUM_CLASSES)),
    },
    "ade20k_val_voc": {
        "source": "ADE20K SceneParsing validation, 150 classes (objectInfo150.txt)",
        "evaluable_classes": sorted({v for _, _, v in ADE20K_TO_VOC if 0 < v < NUM_CLASSES}),
    },
    "cityscapes_val_voc": {
        "source": "Cityscapes val, gtFine labelIds (34 labels)",
        "evaluable_classes": sorted({v for v in CITYSCAPES_TO_VOC.values() if 0 < v < NUM_CLASSES}),
    },
}
for _info in LABEL_REMAPPED_DATASETS.values():  # the extra title line of every figure of these datasets
    _info["title_note"] = (f"GT remapped to VOC labels (merged / ignored classes) · "
                           f"{len(_info['evaluable_classes'])} of {NUM_CLASSES - 1} classes evaluable")


class SegDataset(Dataset):
    """Images and masks given by file paths. Masks are VOC .png or SBD .mat files
    (subclasses without mask files override mask())."""

    def __init__(self, name, image_paths, mask_paths=None):
        self.name = name
        self.image_paths = [Path(p) for p in image_paths]
        self.mask_paths = [Path(p) for p in mask_paths] if mask_paths is not None else None
        self.ids = [p.stem for p in self.image_paths]

    def __len__(self):
        return len(self.ids)

    def image(self, i):
        return Image.open(self.image_paths[i]).convert("RGB")

    def mask(self, i):
        path = self.mask_paths[i]
        if path.suffix == ".mat":  # SBD: no void band, values 0..20
            return loadmat(path)["GTcls"][0]["Segmentation"][0].astype(np.uint8)
        return np.array(Image.open(path), dtype=np.uint8)  # VOC palette png: values 0..20, 255

    def __getitem__(self, i):
        return self.image(i), self.mask(i)


class CocoVOCDataset(SegDataset):
    """COCO images with masks of the 20 VOC categories, built like torchvision's reference
    segmentation scripts (the protocol behind the weights' reported COCO-val numbers):
    annotations of other categories are dropped, each instance mask is the union of its
    polygons / RLE, the class is the VOC index, and pixels covered by more than one
    instance are void (255)."""

    def __init__(self, name, image_dir, ann_file):
        from pycocotools.coco import COCO  # only needed for COCO datasets

        for path in (Path(ann_file), Path(image_dir)):
            if not path.exists():
                raise FileNotFoundError(f"{path} not found (COCO 2017 must be downloaded manually into {COCO_DIR}).")
        self.coco = COCO(str(ann_file))
        self.image_ids = sorted(self.coco.imgs)
        super().__init__(name, [Path(image_dir) / self.coco.imgs[i]["file_name"] for i in self.image_ids])

    def mask(self, i):
        info = self.coco.imgs[self.image_ids[i]]
        target = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        instances = np.zeros_like(target)
        for ann in self.coco.imgToAnns[self.image_ids[i]]:
            voc_class = COCO_TO_VOC.get(ann["category_id"])
            if voc_class is None:
                continue
            instance = self.coco.annToMask(ann)
            target = np.maximum(target, instance * voc_class)
            instances += instance
        target[instances > 1] = VOID_LABEL
        return target

    @property
    def source_classes(self):
        """[(source value, name, VOC class)] of source_mask(): every COCO category of the annotation file."""
        return ([(0, "no annotation", 0)]
                + [(cat_id, cat["name"], COCO_TO_VOC.get(cat_id, 0)) for cat_id, cat in sorted(self.coco.cats.items())]
                + [(VOID_LABEL, "more than one instance (of any category)", VOID_LABEL)])

    def source_mask(self, i):
        """uint8 [H, W]: the COCO category id of every pixel, all 80 categories (for the label check); 0 = no annotation,
        255 = covered by more than one instance of ANY category. (mask() voids only overlaps of VOC instances, so a
        person holding a cup is "more than one instance" here and person in mask().)"""
        info = self.coco.imgs[self.image_ids[i]]
        target = np.zeros((info["height"], info["width"]), dtype=np.uint8)
        instances = np.zeros(target.shape, dtype=np.int32)
        for ann in self.coco.imgToAnns[self.image_ids[i]]:
            instance = self.coco.annToMask(ann)
            target[instance > 0] = ann["category_id"]
            instances += instance
        target[instances > 1] = VOID_LABEL
        return target


class LabelRemappedDataset(SegDataset):
    """Images with label PNGs in another dataset's label space (ADE20K, Cityscapes), remapped to VOC by a 256-entry
    lookup table.
        source_classes  [(source value, source name, VOC class or 255)] for every value the label PNG may hold; any
                        other value maps to 255 (the setup check counts such values)
        inference_size  (width, height) or None (native resolution). Set: the image is resized bilinear and the labels
                        nearest, here, so every later step sees matching sizes."""

    def __init__(self, name, image_paths, mask_paths, source_classes, inference_size=None, ids=None):
        super().__init__(name, image_paths, mask_paths)
        if ids is not None:
            self.ids = list(ids)
        self.source_classes = source_classes
        self.lookup_table = np.full(256, VOID_LABEL, dtype=np.uint8)
        for value, _, voc_class in source_classes:
            self.lookup_table[value] = voc_class
        if inference_size is not None and not (
                len(inference_size) == 2 and all(isinstance(v, int) and not isinstance(v, bool) and v > 0
                                                 for v in inference_size)):
            raise ValueError(f"{name}: the inference size must be None or (width, height) in pixels, got "
                             f"{inference_size!r}")
        self.inference_size = tuple(inference_size) if inference_size is not None else None

    def image(self, i):
        image = super().image(i)
        return image.resize(self.inference_size, Image.BILINEAR) if self.inference_size else image

    def source_mask(self, i):
        labels = Image.open(self.mask_paths[i])
        if self.inference_size:
            labels = labels.resize(self.inference_size, Image.NEAREST)
        labels = np.array(labels)
        if labels.ndim != 2 or labels.dtype != np.uint8:
            raise ValueError(f"{self.mask_paths[i]}: expected one channel of 8-bit labels, got {labels.dtype} "
                             f"{labels.shape}")
        return labels

    def mask(self, i):
        return self.lookup_table[self.source_mask(i)]


def ade20k_source_classes(class_file):
    """[(value, name, VOC class)] of ADE20K-150, from objectInfo150.txt (tab-separated: Idx, Ratio, Train, Val, Name;
    synonyms inside Name separated by ", "). ADE20K_TO_VOC is matched on the exact full Name (whitespace around it
    stripped); each of its names must occur exactly once, at the Idx given there, or this stops. 0 -> 255, every class
    not in ADE20K_TO_VOC -> 0 (background)."""
    names = {}
    with open(class_file) as f:
        header = [field.strip() for field in f.readline().split("\t")]
        if header[0] != "Idx" or header[-1] != "Name":
            raise ValueError(f"{class_file}: unexpected header {header} (expected Idx ... Name)")
        for line in f:
            if line.strip():
                fields = [field.strip() for field in line.rstrip("\n").split("\t")]
                names[int(fields[0])] = fields[-1]
    if sorted(names) != list(range(1, 151)):
        raise ValueError(f"{class_file}: expected the class indices 1-150, got {len(names)} classes")
    indices_of_name = {}
    for idx, name in names.items():
        indices_of_name.setdefault(name, []).append(idx)
    voc_class_of = {}
    for expected_idx, name, voc_class in ADE20K_TO_VOC:
        found = indices_of_name.get(name, [])
        if found != [expected_idx]:
            raise ValueError(f"{class_file}: the class name {name!r} must occur exactly once, at Idx {expected_idx}; "
                             f"found at {found or 'no Idx'}. Check ADE20K_TO_VOC in common/datasets.py.")
        voc_class_of[expected_idx] = voc_class
    return [(0, "other / unlabelled", VOID_LABEL)] + [(idx, names[idx], voc_class_of.get(idx, 0))
                                                      for idx in range(1, 151)]


def _ade20k_val():
    image_dir, label_dir = ADE20K_DIR / "images" / "validation", ADE20K_DIR / "annotations" / "validation"
    class_file = ADE20K_DIR / "objectInfo150.txt"
    for path in (image_dir, label_dir, class_file):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (ADE20K must be downloaded into {ADE20K_DIR.parent}).")
    images = sorted(image_dir.glob("*.jpg"))
    masks = [label_dir / f"{p.stem}.png" for p in images]
    missing = [m.name for m in masks if not m.exists()]
    if not images or missing:
        raise FileNotFoundError(f"{image_dir}: {len(images)} images, {len(missing)} without a label PNG in {label_dir} "
                                f"(e.g. {missing[:3]}).")
    return LabelRemappedDataset("ade20k_val_voc", images, masks, ade20k_source_classes(class_file))


def _cityscapes_val():
    image_dir, label_dir = CITYSCAPES_DIR / "leftImg8bit" / "val", CITYSCAPES_DIR / "gtFine" / "val"
    for path in (image_dir, label_dir):
        if not path.exists():
            raise FileNotFoundError(f"{path} not found (Cityscapes must be downloaded into {CITYSCAPES_DIR}).")
    # sorted, so the image indices are the same on every machine (os.listdir order is not)
    images = sorted(image_dir.glob("*/*_leftImg8bit.png"))
    ids = [p.name[:-len("_leftImg8bit.png")] for p in images]
    masks = [label_dir / p.parent.name / f"{i}_gtFine_labelIds.png" for p, i in zip(images, ids)]
    missing = [m.name for m in masks if not m.exists()]
    if not images or missing:
        raise FileNotFoundError(f"{image_dir}: {len(images)} images, {len(missing)} without a *_gtFine_labelIds.png in "
                                f"{label_dir} (e.g. {missing[:3]}).")
    return LabelRemappedDataset("cityscapes_val_voc", images, masks, CITYSCAPES_SOURCE_CLASSES,
                                C.LABEL_REMAPPED_CITYSCAPES_INFERENCE_SIZE, ids=ids)


def print_mapping_table(dataset, source_images=None, source_pixels=None):
    """Prints where each source class of a label-remapped dataset goes: grouped by target (the VOC object classes, then
    255 ignore, then 0 background), by source value inside a group. With the counts ([256] per source value: images
    that contain it, pixels), also the images, pixels and share of all pixels of each source class."""
    counted = source_pixels is not None
    total = int(np.sum(source_pixels)) if counted else 0
    targets = sorted({v for _, _, v in dataset.source_classes}, key=lambda v: (v in (0, VOID_LABEL), v == 0, v))
    print(f"  {'-> VOC class':18s} {'value':>5s}  {'source class':60s}"
          + (f" {'images':>8s} {'pixels':>15s} {'share':>7s}" if counted else ""))
    for target in targets:
        label = ("0 background" if target == 0 else "255 ignore" if target == VOID_LABEL
                 else f"{target} {VOC_CLASSES[target]}")
        for value, name, voc_class in dataset.source_classes:
            if voc_class != target:
                continue
            row = f"  {label:18s} {value:5d}  {name[:60]:60s}"
            if counted:
                row += (f" {int(source_images[value]):8,d} {int(source_pixels[value]):15,d}"
                        f" {source_pixels[value] / total if total else np.nan:7.2%}")
            print(row)
            label = ""


def _voc(image_set):
    voc = VOCSegmentation(root=VOC_DIR, year="2012", image_set=image_set, download=False)
    return voc.images, voc.masks


def download_missing_voc_and_sbd():
    """Downloads VOC 2012 and SBD, each only if it is not on disk yet (torchvision's SBD download
    moves files out of the archive folder, so repeating it on existing data would fail)."""
    for name, load in (
        ("VOC 2012", lambda download: VOCSegmentation(root=VOC_DIR, year="2012", image_set="train", download=download)),
        ("SBD", lambda download: SBDataset(root=SBD_DIR, image_set="train", mode="segmentation", download=download)),
    ):
        try:
            load(False)
            print(f"  {name}: already on disk, no download.")
        except (RuntimeError, FileNotFoundError):
            print(f"  {name}: not found, downloading ...", flush=True)
            load(True)
            print(f"  {name}: downloaded.")


def _voc_sbd_train():
    """VOC+SBD train: (VOC train ∪ SBD train ∪ SBD val) − VOC val = 10,582 images, each image once.
    VOC annotations are kept for images that are in both VOC and SBD."""
    sources = [_voc("train")]
    for image_set in ("train", "val"):
        sbd = SBDataset(root=SBD_DIR, image_set=image_set, mode="segmentation", download=False)
        sources.append((sbd.images, sbd.masks))
    seen = {Path(p).stem for p in _voc("val")[0]}  # VOC val is held out for evaluation
    images, masks = [], []
    for source_images, source_masks in sources:
        for image, mask in zip(source_images, source_masks):
            if Path(image).stem not in seen:
                seen.add(Path(image).stem)
                images.append(image)
                masks.append(mask)
    return SegDataset("voc_sbd_train", images, masks)


def get_dataset(name):
    if name == "voc_sbd_train":
        return _voc_sbd_train()
    if name == "voc_val":
        return SegDataset("voc_val", *_voc("val"))
    if name in ("coco_train2017_voc", "coco_val2017_voc"):
        split = name.split("_")[1]  # train2017 | val2017
        return CocoVOCDataset(name, COCO_DIR / "images" / split, COCO_DIR / "annotations" / f"instances_{split}.json")
    if name == "ade20k_val_voc":
        return _ade20k_val()
    if name == "cityscapes_val_voc":
        return _cityscapes_val()
    raise ValueError(f"Unknown dataset {name!r}")


class _PixelCounts(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        counts = np.bincount(self.dataset.mask(i).ravel(), minlength=256)[:NUM_CLASSES]
        return torch.from_numpy(counts)


def class_pixel_counts(dataset, num_workers=8):
    """[len(dataset), 21] number of GT pixels of each VOC class per image (void excluded)."""
    loader = DataLoader(_PixelCounts(dataset), batch_size=64, num_workers=num_workers)
    return torch.cat(list(loader)).numpy()
