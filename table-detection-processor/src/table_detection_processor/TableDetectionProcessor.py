from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from nifiapi.relationship import Relationship
import io
import torch
import cv2
import numpy as np
from transformers import TableTransformerForObjectDetection, AutoImageProcessor

class TableDetectionProcessor(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'

    REL_ORIGINAL = Relationship(name="original", description="Original Image")
    REL_ANNOTATED = Relationship(name="annotated", description="Image Annotated")
    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.pretrainedTableModelName = PropertyDescriptor(
            name="table-model-name",
            description="Name of HuggingFace Table Model",
            required = True,
            default_value = "microsoft/table-transformer-structure-recognition-v1.1-all",
            validators = [StandardValidators.NON_EMPTY_VALIDATOR]
        )
        
        self.descriptors = [self.pretrainedTableModelName]

    def onScheduled(self, context):
        modelName = context.getProperty(self.pretrainedTableModelName.name).getValue()
        self.model = TableTransformerForObjectDetection.from_pretrained(modelName)
        self.feature_extractor = AutoImageProcessor.from_pretrained(modelName)

    def plot_results(self, img, scores, labels, boxes):
        colors = COLORS * 255
        for score, label, (xmin, ymin, xmax, ymax),c  in zip(scores.tolist(), labels.tolist(), boxes.tolist(), colors.tolist()):
            self.logger.info('score: ' + str(score) + ' label: ' + str(label) + ' color: ' + str(c) + ' x1: ' + str(xmin) + ' y1: ' + str(ymin) + ' x2: ' + str(xmax) + ' y2: ' + str(ymax)  )
            cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), c, 3)
            text = f'{self.model.config.id2label[label]}: {score:0.2f}'
            font = cv2.FONT_HERSHEY_SIMPLEX
            txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
            cv2.putText(img, text, (int(xmin), int(ymin + txt_size[1])), font, 0.4, c, thickness=1)
        res, img_out = cv2.imencode('.png', img)
        return img_out.tobytes()
    
    def transform(self, context, flowFile):
        
        originalFileNameAtt = flowFile.getAttribute("filename")
        
        image = cv2.imdecode(np.frombuffer(flowFile.getContentsAsBytes(), dtype=np.uint8), cv2.IMREAD_COLOR)
        #image = Image.open(io.BytesIO(flowFile.getContentsAsBytes())).convert("RGB")
        max_size = {}
        max_size['max_height'] = 1000
        max_size['max_width'] = 1000
        encoding = self.feature_extractor(image, return_tensors="pt", size=max_size)
        
        #with torch.no_grad():
        outputs = self.model(**encoding)

        target_sizes = torch.tensor([image.shape[:2]])
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
        outputImage = self.plot_results(image, results['scores'], results['labels'], results['boxes'])

        return FlowFileTransformResult(relationship = "annotated", contents = outputImage)

    def getPropertyDescriptors(self):
        return self.descriptors
    def getRelationships(self):
        return [self.REL_ANNOTATED, self.REL_ORIGINAL]
        
COLORS = np.array(
    [
        0.000, 0.447, 0.741,
        0.850, 0.325, 0.098,
        0.929, 0.694, 0.125,
        0.494, 0.184, 0.556,
        0.466, 0.674, 0.188,
        0.301, 0.745, 0.933,
        0.635, 0.078, 0.184,
        0.300, 0.300, 0.300,
        0.600, 0.600, 0.600,
        1.000, 0.000, 0.000,
        1.000, 0.500, 0.000,
        0.749, 0.749, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 1.000,
        0.667, 0.000, 1.000,
        0.333, 0.333, 0.000,
        0.333, 0.667, 0.000,
        0.333, 1.000, 0.000,
        0.667, 0.333, 0.000,
        0.667, 0.667, 0.000,
        0.667, 1.000, 0.000,
        1.000, 0.333, 0.000,
        1.000, 0.667, 0.000,
        1.000, 1.000, 0.000,
        0.000, 0.333, 0.500,
        0.000, 0.667, 0.500,
        0.000, 1.000, 0.500,
        0.333, 0.000, 0.500,
        0.333, 0.333, 0.500,
        0.333, 0.667, 0.500,
        0.333, 1.000, 0.500,
        0.667, 0.000, 0.500,
        0.667, 0.333, 0.500,
        0.667, 0.667, 0.500,
        0.667, 1.000, 0.500,
        1.000, 0.000, 0.500,
        1.000, 0.333, 0.500,
        1.000, 0.667, 0.500,
        1.000, 1.000, 0.500,
        0.000, 0.333, 1.000,
        0.000, 0.667, 1.000,
        0.000, 1.000, 1.000,
        0.333, 0.000, 1.000,
        0.333, 0.333, 1.000,
        0.333, 0.667, 1.000,
        0.333, 1.000, 1.000,
        0.667, 0.000, 1.000,
        0.667, 0.333, 1.000,
        0.667, 0.667, 1.000,
        0.667, 1.000, 1.000,
        1.000, 0.000, 1.000,
        1.000, 0.333, 1.000,
        1.000, 0.667, 1.000,
        0.333, 0.000, 0.000,
        0.500, 0.000, 0.000,
        0.667, 0.000, 0.000,
        0.833, 0.000, 0.000,
        1.000, 0.000, 0.000,
        0.000, 0.167, 0.000,
        0.000, 0.333, 0.000,
        0.000, 0.500, 0.000,
        0.000, 0.667, 0.000,
        0.000, 0.833, 0.000,
        0.000, 1.000, 0.000,
        0.000, 0.000, 0.167,
        0.000, 0.000, 0.333,
        0.000, 0.000, 0.500,
        0.000, 0.000, 0.667,
        0.000, 0.000, 0.833,
        0.000, 0.000, 1.000,
        0.000, 0.000, 0.000,
        0.143, 0.143, 0.143,
        0.286, 0.286, 0.286,
        0.429, 0.429, 0.429,
        0.571, 0.571, 0.571,
        0.714, 0.714, 0.714,
        0.857, 0.857, 0.857,
        0.000, 0.447, 0.741,
        0.314, 0.717, 0.741,
        0.50, 0.5, 0
    ]
).astype(np.float32).reshape(-1, 3)