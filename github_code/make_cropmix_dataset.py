import shutil
from pathlib import Path
from PIL import Image
import yaml

# ======================
# INPUT (tvoj postojeci YOLO dataset)
# ======================
SRC_BASE = Path(r"C:\Users\zeskinja\Documents\Dario\dataset\dataset\instance_version\yolo_format")
SRC_YAML = SRC_BASE / "trashcan.yaml"

SRC_TR_IMG = SRC_BASE / "train" / "images"
SRC_TR_LBL = SRC_BASE / "train" / "labels"
SRC_VA_IMG = SRC_BASE / "val" / "images"
SRC_VA_LBL = SRC_BASE / "val" / "labels"

# ======================
# OUTPUT (novi cropmix dataset)
# ======================
OUT_BASE = SRC_BASE.parent / "yolo_format_cropmix2"
OUT_TR_IMG = OUT_BASE / "train" / "images"
OUT_TR_LBL = OUT_BASE / "train" / "labels"
OUT_VA_IMG = OUT_BASE / "val" / "images"
OUT_VA_LBL = OUT_BASE / "val" / "labels"

for d in [OUT_TR_IMG, OUT_TR_LBL, OUT_VA_IMG, OUT_VA_LBL]:
    d.mkdir(parents=True, exist_ok=True)

# ======================
# CROP PARAMETRI (tuning)
# ======================
MAX_CROPS_PER_IMAGE = 2   # po tvojoj statistici train avg=1.57 -> 2 je sweet spot
CROP_SCALE = 2.0          # crop je 2x veci od bbox-a (margina)
MIN_BOX_PX = 6            # ignoriraj ekstremno sitne boxeve (u px)
SAVE_QUALITY = 95

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

def read_yolo_labels(lbl_path: Path):
    items = []
    if not lbl_path.exists():
        return items
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            continue
        cls = int(parts[0])
        xc, yc, w, h = map(float, parts[1:])
        items.append((cls, xc, yc, w, h))
    return items

def yolo_to_xyxy(xc, yc, w, h, W, H):
    x1 = (xc - w/2) * W
    y1 = (yc - h/2) * H
    x2 = (xc + w/2) * W
    y2 = (yc + h/2) * H
    return x1, y1, x2, y2

def xyxy_to_yolo(x1, y1, x2, y2, W, H):
    xc = ((x1 + x2) / 2) / W
    yc = ((y1 + y2) / 2) / H
    w  = (x2 - x1) / W
    h  = (y2 - y1) / H
    return xc, yc, w, h

def clamp(v, lo, hi):
    return max(lo, min(hi, v))

def copy_split(src_img_dir, src_lbl_dir, out_img_dir, out_lbl_dir):
    copied_imgs = 0
    copied_lbls = 0
    for img_path in src_img_dir.glob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue
        shutil.copy2(img_path, out_img_dir / img_path.name)
        copied_imgs += 1

        lbl_path = src_lbl_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            shutil.copy2(lbl_path, out_lbl_dir / lbl_path.name)
            copied_lbls += 1
    return copied_imgs, copied_lbls

def add_train_crops():
    crop_count = 0
    skipped_tiny = 0
    for img_path in SRC_TR_IMG.glob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        lbl_path = SRC_TR_LBL / (img_path.stem + ".txt")
        labels = read_yolo_labels(lbl_path)
        if not labels:
            continue

        # UZMI NAJMANJE OBJEKTE (small-object fokus)
        labels = sorted(labels, key=lambda t: t[3] * t[4])  # t=(cls,xc,yc,w,h)
        labels = labels[:MAX_CROPS_PER_IMAGE]

        img = Image.open(img_path).convert("RGB")
        W, H = img.size

        for i, (cls, xc, yc, bw, bh) in enumerate(labels):
            x1, y1, x2, y2 = yolo_to_xyxy(xc, yc, bw, bh, W, H)
            box_w = x2 - x1
            box_h = y2 - y1
            if box_w < MIN_BOX_PX or box_h < MIN_BOX_PX:
                skipped_tiny += 1
                continue

            # proširi crop oko boxa
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            crop_w = box_w * CROP_SCALE
            crop_h = box_h * CROP_SCALE

            cx1 = clamp(cx - crop_w/2, 0, W-1)
            cy1 = clamp(cy - crop_h/2, 0, H-1)
            cx2 = clamp(cx + crop_w/2, 1, W)
            cy2 = clamp(cy + crop_h/2, 1, H)

            crop = img.crop((cx1, cy1, cx2, cy2))
            cW, cH = crop.size

            # bbox unutar cropa
            nx1 = clamp(x1 - cx1, 0, cW)
            ny1 = clamp(y1 - cy1, 0, cH)
            nx2 = clamp(x2 - cx1, 0, cW)
            ny2 = clamp(y2 - cy1, 0, cH)

            nxc, nyc, nw, nh = xyxy_to_yolo(nx1, ny1, nx2, ny2, cW, cH)

            out_stem = f"{img_path.stem}_crop{i}"
            out_img = OUT_TR_IMG / f"{out_stem}.jpg"
            out_lbl = OUT_TR_LBL / f"{out_stem}.txt"

            crop.save(out_img, quality=SAVE_QUALITY)
            out_lbl.write_text(f"{cls} {nxc:.6f} {nyc:.6f} {nw:.6f} {nh:.6f}\n", encoding="utf-8")
            crop_count += 1

    return crop_count, skipped_tiny

def make_yaml():
    # ucitaj original yaml i samo promijeni path
    data = yaml.safe_load(SRC_YAML.read_text(encoding="utf-8"))
    data["path"] = str(OUT_BASE)
    data["train"] = "train/images"
    data["val"] = "val/images"

    out_yaml = OUT_BASE / "trashcan_cropmix2.yaml"
    out_yaml.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_yaml

if __name__ == "__main__":
    print("Copying original train/val into:", OUT_BASE)

    tr_imgs, tr_lbls = copy_split(SRC_TR_IMG, SRC_TR_LBL, OUT_TR_IMG, OUT_TR_LBL)
    va_imgs, va_lbls = copy_split(SRC_VA_IMG, SRC_VA_LBL, OUT_VA_IMG, OUT_VA_LBL)

    print(f"Original copied: train images={tr_imgs}, train labels={tr_lbls}")
    print(f"Original copied:   val images={va_imgs},   val labels={va_lbls}")

    crops, skipped = add_train_crops()
    print(f"Added crop images to TRAIN: {crops} (skipped tiny boxes: {skipped})")

    out_yaml = make_yaml()
    print("? Done.")
    print("New dataset folder:", OUT_BASE)
    print("New YAML:", out_yaml)
