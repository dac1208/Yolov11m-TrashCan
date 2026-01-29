import os
import json
from tqdm import tqdm

def convert_coco_to_yolo(json_path, img_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)

    images = {img['id']: img for img in data['images']}
    categories = {cat['id']: cat for cat in data['categories']}
    cat2idx = {cat_id: idx for idx, cat_id in enumerate(sorted(categories.keys()))}

    annotations_per_image = {}
    for ann in data['annotations']:
        img_id = ann['image_id']
        bbox = ann['bbox']  # [x, y, width, height]
        category_id = ann['category_id']

        if img_id not in annotations_per_image:
            annotations_per_image[img_id] = []

        x, y, w, h = bbox
        x_center = x + w / 2
        y_center = y + h / 2

        img_w = images[img_id].get('width', 1280)
        img_h = images[img_id].get('height', 720)

        x_center /= img_w
        y_center /= img_h
        w /= img_w
        h /= img_h

        annotations_per_image[img_id].append(
            f"{cat2idx[category_id]} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
        )

    # upis u .txt datoteke
    for img_id, lines in tqdm(annotations_per_image.items(), desc=f"Writing labels for {json_path}"):
        filename = images[img_id]['file_name']
        txt_path = os.path.splitext(os.path.join(img_dir, filename))[0] + ".txt"
        with open(txt_path, "w") as f:
            f.write("\n".join(lines))

# Pokreni za oba seta
convert_coco_to_yolo("dataset/instance_version/instances_train_trashcan.json", "dataset/instance_version/train")
convert_coco_to_yolo("dataset/instance_version/instances_val_trashcan.json", "dataset/instance_version/val")
