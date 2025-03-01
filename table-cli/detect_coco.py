import sys
import io
import os
import torch
import cv2
import numpy as np
import json
import logging
from typing import List, Tuple
from coco import CocoDataset, Info, Image, License, Annotation, Category, CocoUtils
from transformers import TableTransformerForObjectDetection, AutoImageProcessor

logger = logging.getLogger("detect_coco")

#Input File
image_file_name = sys.argv[1]
byte_array = []
try:
    with open(image_file_name, 'rb') as file:
        byte_array = bytearray(file.read())
except FileNotFoundError:
    print(f"Error: File not found at path: {file_path}")
    exit()
except Exception as e:
    print(f"An error occurred: {e}")
    exit()

#Load Model
model_name = "microsoft/table-transformer-structure-recognition-v1.1-all"
model = TableTransformerForObjectDetection.from_pretrained(model_name)
feature_extractor = AutoImageProcessor.from_pretrained(model_name)

coco_utils = CocoUtils()

#Convert image to encoding
image = cv2.imdecode(np.frombuffer(byte_array, dtype=np.uint8), cv2.IMREAD_COLOR)
max_size = {}
max_size['max_height'] = 1000
max_size['max_width'] = 1000
encoding = feature_extractor(image, return_tensors="pt", size=max_size)
logger.info('Encoding: ' + str(encoding))

#Execute Model
outputs = model(**encoding)
logger.info('Output: ' + str(outputs))

#Post Process
target_sizes = torch.tensor([image.shape[:2]])
results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
logger.info('Results: ' + str(results))

#Convert to Coco Json
coco_data = coco_utils.build_coco(image_file_name, results['labels'], results['boxes'])
coco_output = coco_data.to_json()

#Output File
os.makedirs("output", exist_ok=True)
with open("output/coco.json", "wb") as binary_file:
    binary_file.write(coco_output.encode("utf-8"))