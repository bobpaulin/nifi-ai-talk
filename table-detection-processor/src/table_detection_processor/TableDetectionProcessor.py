from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.relationship import Relationship
import io
import torch
import cv2
import numpy as np
import json
from typing import List, Tuple
from coco import CocoDataset, Info, Image, License, Annotation, Category, CocoUtils
from plot_utils import PlotUtils
from transformers import TableTransformerForObjectDetection, AutoImageProcessor

class TableDetectionProcessor(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'

    REL_COCO = Relationship(name="coco", description="Coco Data")
    REL_ANNOTATED = Relationship(name="annotated", description="Image Annotated")
    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.pretrained_table_model_name = PropertyDescriptor(
            name="Table Model Name",
            description="Name of HuggingFace Table Model",
            required = True,
            default_value = "microsoft/table-transformer-structure-recognition-v1.1-all",
            validators = [StandardValidators.NON_EMPTY_VALIDATOR]
        )
        
        self.output_type = PropertyDescriptor(
            name="Output Type",
            description="Output Type (Coco or Annotated Image)",
            required = True,
            allowable_values = ["coco", "annotations"],
            default_value = "annotations",
        )
        
        self.descriptors = [self.pretrained_table_model_name, self.output_type]

    def getPropertyDescriptors(self):
        return self.descriptors
    def getRelationships(self):
        return [self.REL_ANNOTATED, self.REL_COCO]

    def onScheduled(self, context):
        model_name = context.getProperty(self.pretrained_table_model_name.name).getValue()
        
        #Load Model
        self.model = TableTransformerForObjectDetection.from_pretrained(model_name)
        self.feature_extractor = AutoImageProcessor.from_pretrained(model_name)
        self.plot_utils = PlotUtils(self.model, self.logger)
        self.coco_utils = CocoUtils()

    def transform(self, context, flow_file):
        
        original_file_name_att = flow_file.getAttribute("filename")
        original_attributes = flow_file.getAttributes()

        #Convert image to encoding
        image = cv2.imdecode(np.frombuffer(flow_file.getContentsAsBytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        max_size = {}
        max_size['max_height'] = 1000
        max_size['max_width'] = 1000
        encoding = self.feature_extractor(image, return_tensors="pt", size=max_size)
        self.logger.info('Encoding: ' + str(encoding))

        #Execute Model
        outputs = self.model(**encoding)
        self.logger.info('Output: ' + str(outputs))

        #Post Process
        target_sizes = torch.tensor([image.shape[:2]])
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
        self.logger.info('Results: ' + str(results))

        selected_output_type = context.getProperty(self.output_type.name).getValue()
        if selected_output_type == 'coco':
            #Convert to Coco Json
            coco_data = self.coco_utils.build_coco(original_file_name_att, results['labels'], results['boxes'])
            coco_output = coco_data.to_json()
            update_attributes = {"mime.type": "application/json", "filename": original_file_name_att + ".json", "original-filename": original_file_name_att}
            return FlowFileTransformResult(relationship = "coco", contents = coco_output, attributes = update_attributes)
        else:
            #Write Detected Image
            output_image = self.plot_utils.plot_results(image, results['scores'], results['labels'], results['boxes'])
            return FlowFileTransformResult(relationship = "annotated", contents = output_image)
