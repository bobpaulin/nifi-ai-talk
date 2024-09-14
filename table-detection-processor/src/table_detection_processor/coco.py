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