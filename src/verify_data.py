# .\src\verify_data.py --yolo-dir .\data\ --crops-dir .\cropped\

import argparse
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


def color_for_class(class_id):
    return PALETTE_RGB[class_id % len(PALETTE_RGB)]

def load_classes(classes_file):
    classes = {}
    if not classes_file.exists():
        return classes
    with open(classes_file, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) == 2 and parts[0].isdigit():
                classes[int(parts[0])] = parts[1]
            else:
                classes[i] = line
    return classes


def find_image_files(images_dir):
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    return [p for p in images_dir.rglob("*") if p.suffix.lower() in exts]


def parse_yolo_label(label_path):
    boxes = []
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                xc, yc, w, h = map(float, parts[1:5])
            except ValueError:
                continue
            boxes.append((class_id, xc, yc, w, h))
    return boxes


def draw_yolo_boxes(img_bgr, boxes, classes):
    h, w = img_bgr.shape[:2]
    img_out = img_bgr.copy()
    for (class_id, xc, yc, bw, bh) in boxes:
        x1 = max(0, int((xc - bw / 2) * w))
        y1 = max(0, int((yc - bh / 2) * h))
        x2 = min(w - 1, int((xc + bw / 2) * w))
        y2 = min(h - 1, int((yc + bh / 2) * h))
        r, g, b = color_for_class(class_id)
        color_bgr = (int(b * 255), int(g * 255), int(r * 255))
        cv2.rectangle(img_out, (x1, y1), (x2, y2), color_bgr, thickness=2)
        label = classes.get(class_id, str(class_id))
        font_scale = max(0.5, min(w, h) / 800)
        thickness_txt = max(1, int(font_scale * 2))
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness_txt)
        bg_y1 = max(0, y1 - th - baseline - 4)
        cv2.rectangle(img_out, (x1, bg_y1), (x1 + tw + 4, y1), color_bgr, cv2.FILLED)
        cv2.putText(img_out, label, (x1 + 2, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness_txt, cv2.LINE_AA)
    return img_out


def verify_yolo():
    images_dir = YOLO_DIR / "images"
    labels_dir = YOLO_DIR / "labels"
    classes_file = YOLO_DIR / "classes.txt"

    if not images_dir.exists():
        for sub in ["train", "val", "test"]:
            if (YOLO_DIR / "images" / sub).exists():
                images_dir = YOLO_DIR / "images" / sub
                labels_dir = YOLO_DIR / "labels" / sub
                break

    classes = load_classes(classes_file)
    all_images = find_image_files(images_dir)

    labeled = []
    for img in all_images:
        label_path = labels_dir / (img.stem + ".txt")
        if label_path.exists():
            labeled.append((img, label_path))

    sample = random.sample(labeled, min(N, len(labeled)))

    fig, axes = plt.subplots(1, len(sample), figsize=(5 * len(sample), 5))
    if len(sample) == 1:
        axes = [axes]
    fig.suptitle("YOLO bounding boxes verification", fontsize=13, fontweight="bold")

    for ax, (img_path, label_path) in zip(axes, sample):
        img_bgr = cv2.imread(str(img_path))
        boxes = parse_yolo_label(label_path)
        img_drawn = draw_yolo_boxes(img_bgr, boxes, classes)
        ax.imshow(cv2.cvtColor(img_drawn, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{img_path.name}\n{len(boxes)} object(s)", fontsize=8)
        ax.axis("off")

    if classes:
        patches = [mpatches.Patch(color=color_for_class(cid), label=f"{cid}: {name}")
                   for cid, name in sorted(classes.items())]
        fig.legend(handles=patches, loc="lower center", ncol=len(classes),
                   bbox_to_anchor=(0.5, -0.04), fontsize=9)

    plt.tight_layout()
    plt.savefig("verify_yolo.png", dpi=150, bbox_inches="tight")
    plt.show()


def verify_crops():
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    all_crops = []
    for class_dir in sorted(CROPS_DIR.iterdir()):
        if class_dir.is_dir():
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in exts:
                    all_crops.append((img_path, class_dir.name))

    sample = random.sample(all_crops, min(N, len(all_crops)))

    fig, axes = plt.subplots(1, len(sample), figsize=(4 * len(sample), 4))
    if len(sample) == 1:
        axes = [axes]
    fig.suptitle("MobileNet crops verification", fontsize=13, fontweight="bold")

    class_names = sorted(set(c for _, c in all_crops))
    for ax, (img_path, class_name) in zip(axes, sample):
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]
        ax.imshow(img_rgb)
        ax.set_title(f"{class_name}\n{w}x{h} px", fontsize=9, fontweight="bold")
        ax.axis("off")
        class_idx = class_names.index(class_name)
        for spine in ax.spines.values():
            spine.set_edgecolor(color_for_class(class_idx))
            spine.set_linewidth(3)
            spine.set_visible(True)

    plt.tight_layout()
    plt.savefig("verify_crops.png", dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-dir", type=Path, default=Path("yolo_dataset"))
    parser.add_argument("--crops-dir", type=Path, default=Path("crops_dataset"))
    args = parser.parse_args()

    YOLO_DIR = args.yolo_dir
    CROPS_DIR = args.crops_dir
    
    if not YOLO_DIR.exists():
        print(f"Directory: {YOLO_DIR} not found.")
        exit(1)
        
    elif not CROPS_DIR.exists():
        print(f"Directory: {CROPS_DIR} not found.")
        exit(1)
    
    N = 5

    PALETTE_RGB = [
        (0.93, 0.17, 0.17),
        (0.17, 0.63, 0.93),
        (0.17, 0.93, 0.40),
        (0.97, 0.76, 0.08),
        (0.70, 0.17, 0.93),
        (0.93, 0.50, 0.17),
        (0.17, 0.93, 0.93),
        (0.93, 0.17, 0.70),
    ]
    

    verify_yolo()
    verify_crops()