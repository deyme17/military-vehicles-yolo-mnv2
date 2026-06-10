import json
from pathlib import Path
import argparse
from tqdm import tqdm



class CocoToYolo:
    def __init__(self, coco: Path, out: Path, cls: Path):
        self.out = out
        self.cls = cls
        self.data = self.__coco_verify(coco)

    def __coco_verify(self, coco: Path) -> dict:
        """
        Loading the json file and verifying that it is a coco file.

        Args:
            coco (Path): Path to the coco file

        Returns:
            dict: json file as a dictionary
        """

        try:
            with open(coco, 'r', encoding="UTF-8") as f:
                data = json.load(f)
        except Exception as e:
            raise RuntimeError(f"An error occurred while opening or loading the coco file:\n{e}") from e

        #Verification that the json in the coco format
        if list(data.keys()) != ['info', 'licenses', 'categories', 'images', 'annotations']:
            raise ValueError(f"{coco} is not a coco file")

        return data

    def __get_categories(self) -> dict:
        """
        Retrieve a category dictionary in the id: name format from the list of dictionaries.

        Returns: 
            dict: A dictionary in format {id: name}
        """
        cats = self.data.get("categories")
        
        for i, cat in enumerate(cats):
            idx = cat.get("id", "")
            name = cat.get("name", "")
            
            if idx == "" or name == "":
                raise ValueError(f"The element {i} in the categories does not have an id or name attribute\n{cat}")
        
        #Filtering supercategories
        real_cats = [c for c in cats if c.get("supercategory") != "none"]
        res = {c["id"]: i for i, c in enumerate(real_cats)}
        
        return res

    def __get_images(self) -> dict:
        """
        Retrieve key data from the list of image dictionaries.

        Returns:
            dict: a dictionary mapping image id's to their data.  
                The inner dict contains:  
                - file_name (str): The name of the file  
                - height (int): The height of the image  
                - width (int): The width of the image  
        """
        res = {}
        imgs = self.data.get("images")

        for i, img in enumerate(imgs):
            idx = img.get("id", "")
            f_name = img.get("file_name", "")
            height = img.get("height", 0)
            width = img.get("width", 0)

            if idx == "" or f_name == "" or height <= 0 or width <= 0:
                raise ValueError(f"The element {i} in the images does not have an id, name, height or width attribute\n{img}")
            
            res[idx] = {
                "file_name": f_name,
                "height": height,
                "width": width
            }
            
        return res

    def __annot_verify(self) -> list:
        """
        Checking for the existence of parameters in each annotation dictionary.

        Returns:
            list: list of annotation dictionaries
        """
        annot_list = self.data.get("annotations")
        res = []

        for i, annot in enumerate(annot_list):
            #Skip images where the iscrowd parameter != 0
            if annot.get("iscrowd", 0) != 0:
                continue
            img_id = annot.get("image_id", "")
            cat_id = annot.get("category_id", "")
            bbox = annot.get("bbox", [0, 0, 0, 0])

            if img_id == "" or cat_id == "":
                raise ValueError(f"The element {i} in the annotations does not have an image_id or category_id attribute\n{annot}")
            
            if any(v < 0 for v in bbox[:2]) or any(v <= 0 for v in bbox[2:]) or len(bbox) != 4:
                raise ValueError(f"The bbox of the element {i} in the annotations has value <= 0 or len != 4 \n{annot}")
            
            res.append(annot)
        
        return res

    def __yolo_calc(self, bbox: list, img_h: int, img_w: int) -> list:
        """
        Converting Coco coordinates to YOLO.

        Args:
            bbox (list): list of coco coordinates
            img_h (int): original image height
            img_w (int): original image width
        
        Returns:
            list: List of strings with yolo float coordinates
        """
        w = bbox[2]
        h = bbox[3]
        
        cx = (bbox[0] + w/2) / img_w
        cy = (bbox[1] + h/2) / img_h
        w_norm = w / img_w
        h_norm = h / img_h
        
        if any(coord > 1 or coord < 0 for coord in [cx, cy, w_norm, h_norm]):
            raise ValueError("The coordinates after transformation are outside the range (0, 1)")

        return [str(cx), str(cy), str(w_norm), str(h_norm)]

    def convert(self) -> None:
        """
        Converting coco files to yolo.
        """
        categories = self.__get_categories()
        images = self.__get_images()
        annot_list = self.__annot_verify()

        for annot in tqdm(annot_list):
            img = images.get(annot.get("image_id"))
            cat_id = categories.get(annot.get("category_id"))
            img_name = img.get("file_name")
            img_w = img.get("width")
            img_h = img.get("height")
            bbox = annot.get("bbox")

            yolo_coord = self.__yolo_calc(bbox, img_h, img_w)
            yolo_str = ' '.join(yolo_coord)
            yolo_name = '.'.join(img_name.split('.')[:-1]) + ".txt"

            with open(self.out.joinpath(yolo_name), 'a', encoding="UTF-8") as f:
                f.write(f"{cat_id} {yolo_str}\n")
                
    
    def create_classes(self):
        """
        Creates a file classes.txt in the output directory
        """
        cats = self.data.get("categories")
        cats_map = self.__get_categories()

        with open(self.cls.joinpath("classes.txt"), 'w', encoding="UTF-8") as f:
            for cat in cats:
                if cat.get("supercategory") == "none":
                    continue
                idx = cats_map.get(cat.get("id"))
                name = cat.get("name")
                f.write(f"{idx} - {name}\n")



def rec_mkdir(d: Path) -> None:
    """
    Recursively creates directories for a specified path

    Args:
        d (Path): directory path
    """
    par = d.parent
    if not par.exists():
        rec_mkdir(par)
    d.mkdir()



# main section
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="coco to YOLOv11 converter")

    parser.add_argument("-f", "--file", type=str, help="Path to coco file")
    parser.add_argument("-o", "--output", type=str, help="Output directory for YOLO files")
    parser.add_argument("-c", "--classes", type=str, default=None, 
                        help="Output directory for classes.txt (-o is used if not defined)")

    args = parser.parse_args()
    
    if args.file is None or args.output is None:
        raise ValueError("The path to coco or YOLO is not specified\nTry --help")

    coco = Path(args.file)
    out = Path(args.output)
    cls = Path(args.classes) if args.classes else out
    
    if coco.is_file():
        ext = (coco.name).split('.')[-1]
        if ext != "json":
            raise ValueError(f"{str(coco)} is not a json")
    else:
        raise FileNotFoundError(f"{str(coco)} is not a file")

    for d in (out, cls):
        if d.exists():
            if not d.is_dir():
                raise NotADirectoryError(f"Path: {d} is not a directory")
        else:
            if input(f"Path {d} doesn't exists, do you want to create this directories? (y/n)\n") != 'y':
                exit(1)
            try:
                d.mkdir()
            except FileNotFoundError:
                rec_mkdir(d)
    
    converter = CocoToYolo(coco, out, cls)

    converter.convert()
    converter.create_classes()