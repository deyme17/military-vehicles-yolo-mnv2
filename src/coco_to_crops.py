import argparse
import json
import logging
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from PIL import Image
from tqdm import tqdm


class OutOfBoundsError(Exception):
    """Raised when a square crop cannot fit inside image bounds."""


INVALID_PATH_CHARS = re.compile(r'[<>:"/\\|?*]')


def sanitize_name(name: str) -> str:
    return INVALID_PATH_CHARS.sub("_", name).strip()


def validate_coco(data: dict) -> None:
    required = {"images", "annotations", "categories"}
    missing = required - set(data.keys())
    if missing:
        raise ValueError(f"Missing COCO fields: {', '.join(sorted(missing))}")


def crop_sample(
    image_path: str | Path,
    result_path: str | Path,
    bbox: list[float],
    margin: float,
    strict_margin: bool = True,
):
    if margin < 0:
        raise ValueError("Margin must be non-negative")
    
    image_path = Path(image_path)
    result_path = Path(result_path)

    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    with Image.open(image_path) as img:
        img_w, img_h = img.size

        x_min, y_min, w, h = bbox
        x_center = x_min + w / 2
        y_center = y_min + h / 2

        base_size = max(w, h)
        requested_size = base_size * (1.0 + margin)

        if strict_margin:
            size = requested_size
            
        else:
            max_dist_x = min(x_center, img_w - x_center)
            max_dist_y = min(y_center, img_h - y_center)
            max_size = min(max_dist_x, max_dist_y) * 2

            if max_size < base_size:
                raise OutOfBoundsError(f"Cannot create square crop for {image_path}")

            size = min(requested_size, max_size)

        x1 = x_center - size / 2
        y1 = y_center - size / 2
        x2 = x_center + size / 2
        y2 = y_center + size / 2

        if x1 < 0 or y1 < 0 or x2 > img_w or y2 > img_h:
            raise OutOfBoundsError(f"Crop exceeds image bounds: {image_path}")

        cropped = img.crop((int(x1), int(y1), int(x2), int(y2)))

        result_path.parent.mkdir(parents= True, exist_ok= True)

        save_kwargs = {}
        if result_path.suffix.lower() in {".jpg", ".jpeg"}:
            cropped = cropped.convert("RGB")
            save_kwargs["quality"] = 95

        cropped.save(result_path, **save_kwargs)
        return True


def build_tasks(
    data: dict,
    images_dir: Path,
    output_dir: Path,
    margin: float,
    output_format: str | None,
    strict_margin: bool,
) -> list[dict]:
    categories = {c["id"]: sanitize_name(c["name"]) for c in data["categories"]}
    images = {img["id"]: img for img in data["images"]}

    image_class_counts = {}

    tasks = []
    for annotation in data["annotations"]:
        if annotation.get("iscrowd", 0):
            continue

        image_info = images.get(annotation["image_id"])
        if image_info is None:
            logging.warning("Missing image_id= %s", annotation["image_id"])
            continue

        class_name = categories.get(annotation["category_id"])
        if class_name is None:
            logging.warning("Missing category_id= %s", annotation["category_id"])
            continue

        image_path = images_dir / image_info["file_name"]
        stem = Path(image_info["file_name"]).stem

        key = (stem, class_name)
        image_class_counts[key] = image_class_counts.get(key, 0) + 1
        crop_index = image_class_counts[key] - 1

        ext = f".{output_format.lstrip('.')}" if output_format else image_path.suffix

        result_path = output_dir / class_name / f"{stem}_crop_{crop_index:04d}{ext}"

        tasks.append({
            "image_path": image_path,
            "result_path": result_path,
            "bbox": annotation["bbox"],
            "margin": margin,
            "strict_margin": strict_margin,
        })

    return tasks


def coco_to_crops(
    json_path: str | Path,
    images_dir: str | Path,
    output_dir: str | Path,
    margin: float = 0.2,
    parallel: bool = True,
    max_workers: int | None = None,
    output_format: str | None = None,
    strict_margin: bool = True,
):
    if margin < 0:
        logging.error("Margin must be non-negative")
        return
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    validate_coco(data)

    tasks = build_tasks(
        data= data,
        images_dir= Path(images_dir),
        output_dir= Path(output_dir),
        margin= margin,
        output_format= output_format,
        strict_margin= strict_margin,
    )
    
    total_tasks = len(tasks)
    success_count = 0
    skipped_count = 0

    with tqdm(total= total_tasks, desc= "Cropping Progress") as pbar:
        if parallel and total_tasks > 1:
            with ProcessPoolExecutor(max_workers= max_workers) as executor:
                futures = [
                    executor.submit(crop_sample, **task_kwargs)
                    for task_kwargs in tasks
                ]

                for future in as_completed(futures):
                    try:
                        if future.result():
                            success_count += 1

                    except OutOfBoundsError as e:
                        skipped_count += 1
                        logging.warning(str(e))

                    except Exception as e:
                        logging.error(str(e))

                    pbar.update(1)
                    
        else:
            for task_kwargs in tasks:
                try:
                    if crop_sample(**task_kwargs):
                        success_count += 1
                        
                except OutOfBoundsError as e:
                    skipped_count += 1
                    logging.warning(str(e))
                    
                except Exception as e:
                    logging.error(str(e))
                    
                pbar.update(1)

    summary = f"Processed {success_count}/{len(tasks)} crops. Skipped non-square: {skipped_count}. Saved to: {output_dir}"
    logging.info(summary)
    print(summary)


if __name__ == "__main__":
    
    logging_format = "%(asctime)s [%(levelname)s]: %(message)s"

    logging.basicConfig(
        filename= "cropping.log",
        filemode= "w",
        format= logging_format,
        level= logging.INFO
    )

    parser = argparse.ArgumentParser(description= "Convert COCO bounding boxes into square crops.")

    parser.add_argument("--json-path", default= "dataset/_annotations.coco.json")
    parser.add_argument("--images-dir", default= "dataset")
    parser.add_argument("--output-dir", default= "cropped")
    parser.add_argument("--margin", type= float, default= 0.20)
    parser.add_argument("--serial", action= "store_true")
    parser.add_argument("--workers", type= int, default= None)
    parser.add_argument("--format", default= None)
    parser.add_argument("--no-strict-margin", action= "store_false", dest= "strict_margin")

    args = parser.parse_args()

    coco_to_crops(
        json_path= args.json_path,
        images_dir= args.images_dir,
        output_dir= args.output_dir,
        margin= args.margin,
        parallel= not args.serial,
        max_workers= args.workers,
        output_format= args.format,
        strict_margin= args.strict_margin,
    )