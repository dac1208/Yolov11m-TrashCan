import shutil
from pathlib import Path
import yaml

SRC_BASE = Path(r"C:\Users\zeskinja\Documents\Dario\dataset\dataset\instance_version\yolo_format")
OUT_BASE = SRC_BASE.parent / "yolo_format_3cls"

splits = ["train", "val"]

def ensure_dirs():
    for s in splits:
        (OUT_BASE / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT_BASE / s / "labels").mkdir(parents=True, exist_ok=True)

def copy_images():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    for s in splits:
        src_img = SRC_BASE / s / "images"
        out_img = OUT_BASE / s / "images"
        n = 0
        for p in src_img.glob("*"):
            if p.suffix.lower() in exts:
                shutil.copy2(p, out_img / p.name)
                n += 1
        print(f"Copied {n} images for split={s}")

def map_class(old_id: int) -> int:
    # 0 = rov, 1 = bio, 2 = trash
    if old_id == 0:
        return 0
    if 1 <= old_id <= 7:
        return 1
    if 8 <= old_id <= 21:
        return 2
    raise ValueError(f"Unexpected class id: {old_id}")

def remap_labels():
    for s in splits:
        src_lbl = SRC_BASE / s / "labels"
        out_lbl = OUT_BASE / s / "labels"
        files = list(src_lbl.glob("*.txt"))
        print(f"Remapping {len(files)} label files for split={s}")

        for fp in files:
            lines_out = []
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    continue
                old_id = int(parts[0])
                new_id = map_class(old_id)
                lines_out.append(" ".join([str(new_id)] + parts[1:]))

            (out_lbl / fp.name).write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")

def make_yaml():
    data = {
        "path": str(OUT_BASE),
        "train": "train/images",
        "val": "val/images",
        "names": {
            0: "rov",
            1: "bio",
            2: "trash"
        }
    }
    out_yaml = OUT_BASE / "trashcan_3cls.yaml"
    out_yaml.write_text(yaml.dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print("Wrote YAML:", out_yaml)

if __name__ == "__main__":
    ensure_dirs()
    copy_images()
    remap_labels()
    make_yaml()
    print("? Done. New dataset:", OUT_BASE)
