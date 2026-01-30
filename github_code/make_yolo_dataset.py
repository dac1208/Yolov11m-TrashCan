import os, json, shutil
from tqdm import tqdm
from PIL import Image
import yaml

ROOT = os.path.dirname(__file__)
BASE = os.path.join(ROOT, "dataset", "dataset", "instance_version")

train_json = os.path.join(BASE, "instances_train_trashcan.json")
val_json   = os.path.join(BASE, "instances_val_trashcan.json")
train_img_dir = os.path.join(BASE, "train")
val_img_dir   = os.path.join(BASE, "val")

yolo_base = os.path.join(BASE, "yolo_format")
ytr_img  = os.path.join(yolo_base, "train", "images")
ytr_lbl  = os.path.join(yolo_base, "train", "labels")
yval_img = os.path.join(yolo_base, "val", "images")
yval_lbl = os.path.join(yolo_base, "val", "labels")
for d in [ytr_img, ytr_lbl, yval_img, yval_lbl]:
    os.makedirs(d, exist_ok=True)

def convert_split(coco_json, src_img_dir, dst_img_dir, dst_lbl_dir):
    with open(coco_json, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = {img["id"]: img for img in coco["images"]}
    anns_by_img = {}
    for ann in coco["annotations"]:
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    categories = {cat["id"]: cat["name"] for cat in coco["categories"]}

    for img_id, img_info in tqdm(images.items(), desc=f"COCO?YOLO {os.path.basename(coco_json)}"):
        file_name = img_info["file_name"]
        src = os.path.join(src_img_dir, file_name)
        if not os.path.exists(src):
            continue

        shutil.copy2(src, os.path.join(dst_img_dir, file_name))

        label_path = os.path.join(dst_lbl_dir, os.path.splitext(file_name)[0] + ".txt")
        lines = []

        W = img_info.get("width")
        H = img_info.get("height")
        if not (W and H):
            with Image.open(src) as im:
                W, H = im.size

        for ann in anns_by_img.get(img_id, []):
            cid = ann["category_id"] - 1  # COCO id obicno krece od 1
            x, y, w, h = ann["bbox"]
            xc = (x + w/2) / W
            yc = (y + h/2) / H
            ww = w / W
            hh = h / H

            xc = min(max(xc, 0.0), 1.0)
            yc = min(max(yc, 0.0), 1.0)
            ww = min(max(ww, 0.0), 1.0)
            hh = min(max(hh, 0.0), 1.0)
            if ww <= 0 or hh <= 0:
                continue

            lines.append(f"{cid} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}")

        with open(label_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    return categories

cats_tr = convert_split(train_json, train_img_dir, ytr_img, ytr_lbl)
convert_split(val_json,   val_img_dir,   yval_img, yval_lbl)

yaml_path = os.path.join(yolo_base, "trashcan.yaml")
names = list(cats_tr.values())
dataset_yaml = {
    "path": yolo_base,
    "train": "train/images",
    "val": "val/images",
    "names": {i: n for i, n in enumerate(names)}
}
with open(yaml_path, "w", encoding="utf-8") as f:
    yaml.dump(dataset_yaml, f, sort_keys=False)

print("? YOLO dataset:", yolo_base)
print("? YAML:", yaml_path)
print("? classes:", names)
