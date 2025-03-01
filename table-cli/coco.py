from typing import List, Tuple
from dataclasses import dataclass
from dataclasses_json import dataclass_json

@dataclass_json
@dataclass
class Info:
    year: int
    version: str
    description: str

@dataclass_json
@dataclass
class Image:
    id: int
    file_name: str

@dataclass_json
@dataclass
class License:
    id: int
    name: str

@dataclass_json
@dataclass
class Annotation:
    id: int
    image_id: int
    category_id: int
    bbox: Tuple[float, float, float, float]
    segmentation: List[List[float]]

@dataclass_json
@dataclass
class Category:
    id: int
    name: str
    supercategory: str

@dataclass_json
@dataclass
class CocoDataset:
    info: Info
    images: List[Image]
    licenses: List[License]
    annotations: List[Annotation]
    categories: List[Category]

class CocoUtils:
    def build_coco(self, filename, labels, boxes):
        info = Info(
            year='2024',
            version = '1.0',
            description = 'A coco file',
        )
        images = [Image(
            id=0,
            file_name=filename,
        )]
        license  = License(
            id=0,
            name="Apache",
        )
        categories = [Category(
            id=0,
            name="table",
            supercategory="table",
        ),
        Category(
            id=1,
            name="table column",
            supercategory="table column",
        ),
        Category(
            id=2,
            name="table row",
            supercategory="table row",
        ),
        Category(
            id=3,
            name="table column header",
            supercategory="table column header",
        ),
        Category(
            id=4,
            name="table projected row header",
            supercategory="table projected row header",
        ),
        Category(
            id=5,
            name="table spanning cell",
            supercategory="table spanning cell",
        ),
        ]
        annotations = []
        for label, (xmin, ymin, xmax, ymax) in zip(labels.tolist(), boxes.tolist()):
            segmentation = []
            segmentation.append((int(xmin), int(ymin), int(xmax), int(ymin), int(xmax), int(ymax), int(xmin), int(ymax)))
            annotations.append(Annotation(
                id=len(annotations),
                image_id=0,
                category_id=label,
                bbox=(int(xmin), int(ymin), int((xmax-xmin)), int((ymax-ymin))),
                segmentation=segmentation
            ))
        # Create the dataset
        return CocoDataset(  
            info=info,
            images=images,
            licenses=[license],
            categories=categories,
            annotations=annotations
        )