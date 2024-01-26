import pandas as pd
from PIL import Image
import json

def csv_to_coco(route_csv: str, route_training_img: str, output_json: str):
    json_pd = pd.read_csv(route_csv)

    info = dict()
    info["year"] = 2024
    info["version"] = "1"
    info["description"] = "COCO 2024 Dataset"
    info["contributor"] = "COCO Consortium"
    info["url"] = "http://cocodataset.org"
    info["date_created"] = "2024-01-15T00:00:00+00:00"

    licenses = list()
    licenses.append({"id": 1, "name": "Attribution-NonCommercial-ShareAlike License",
                     "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"})

    categories = list()
    categories.append({"id": 0, "name": "cars", "supercategory": "none"})

    images = list()
    img_names = json_pd.image.unique().tolist()

    for id_data, img_name in enumerate(img_names):
        image = dict()
        image["id"] = id_data
        image["license"] = 1
        image["file_name"] = img_name
        with Image.open(f"{route_training_img}/{img_name}") as img:
            image["height"] = img.height
            image["width"] = img.width
        image["date_captured"] = "2024-01-15T00:00:00+00:00"
        images.append(image)

    annotations = list()
    annotation_id = 0

    for id_data, img_name in enumerate(img_names):

        img_info = json_pd[json_pd.image == img_name].to_dict(orient='records')
        for img in img_info:
            annotation = dict()
            annotation["id"] = annotation_id
            annotation["image_id"] = id_data
            annotation["category_id"] = 0
            x_min = img["xmin"]
            y_min = img["ymin"]
            width = img["xmax"] - x_min
            height = img["ymax"] - y_min
            annotation["bbox"] = [x_min, y_min, width, height]
            annotation["area"] = width * height
            annotation["segmentation"] = []
            annotation["iscrowd"] = 0
            annotation_id += 1
            annotations.append(annotation)

    coco_format_json = dict()
    coco_format_json["info"] = info
    coco_format_json["licenses"] = licenses
    coco_format_json["categories"] = categories
    coco_format_json["images"] = images
    coco_format_json["annotations"] = annotations

    with open(output_json, 'w') as json_file:
        json.dump(coco_format_json, json_file, indent=4)

    return f"JSON file saved in {output_json}"
