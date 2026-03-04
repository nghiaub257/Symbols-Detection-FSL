import os
import json
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog

def load_symbols_json(json_file, image_root, metadata):
    with open(json_file, 'r') as f:
        coco_data = json.load(f)
    
    img_id_to_imgs = {img['id']: img for img in coco_data['images']}
    img_id_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_id_to_anns:
            img_id_to_anns[img_id] = []
        img_id_to_anns[img_id].append(ann)
    
    id_map = metadata.get("thing_dataset_id_to_contiguous_id", None)
    
    dataset_dicts = []
    for img_id, img_info in img_id_to_imgs.items():
        record = {
            "file_name": os.path.join(image_root, img_info["file_name"]),
            "height": img_info["height"],
            "width": img_info["width"],
            "image_id": img_id,
            "annotations": []
        }
        
        anns = img_id_to_anns.get(img_id, [])
        for ann in anns:
            obj = {
                "bbox": ann["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": ann["category_id"],
                "iscrowd": ann.get("iscrowd", 0)
            }
            if id_map:
                if obj["category_id"] in id_map:
                    obj["category_id"] = id_map[obj["category_id"]]
                    record["annotations"].append(obj)
            else:
                record["annotations"].append(obj)
        
        if record["annotations"]:
            dataset_dicts.append(record)
            
    return dataset_dicts

def register_meta_symbols(name, metadata, imgdir, annofile):
    DatasetCatalog.register(
        name,
        lambda: load_symbols_json(annofile, imgdir, metadata),
    )

    if "_base" in name or "_novel" in name:
        split = "base" if "_base" in name else "novel"
        metadata["thing_dataset_id_to_contiguous_id"] = metadata[
            "{}_dataset_id_to_contiguous_id".format(split)
        ]
        metadata["thing_classes"] = metadata["{}_classes".format(split)]

    MetadataCatalog.get(name).set(
        json_file=annofile,
        image_root=imgdir,
        evaluator_type="coco",
        **metadata,
    )
