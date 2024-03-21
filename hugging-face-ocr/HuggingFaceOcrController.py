from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io

class HuggingFaceOcrController(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'


    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.pretrainedOcrModel = PropertyDescriptor(
            name="ocr-model-name",
            description="Name of HuggingFace OCR Model",
            required = True,
            default_value = "microsoft/trocr-large-handwritten",
            validators = [StandardValidators.NON_EMPTY_VALIDATOR]
        )
        
        self.outputAttribute = PropertyDescriptor(
            name="output-attribute",
            description="Attribute to write Model Output To",
            required = False,
            validators = [StandardValidators.NON_EMPTY_VALIDATOR]
        )
        
        self.descriptors = [self.pretrainedOcrModel, self.outputAttribute]

    def onScheduled(self, context):
        modelName = context.getProperty(self.pretrainedOcrModel.name).getValue()
        self.processor = TrOCRProcessor.from_pretrained(modelName)
        self.model = VisionEncoderDecoderModel.from_pretrained(modelName)
       

    def transform(self, context, flowFile):
        
        outputAttributeName = context.getProperty(self.outputAttribute.name).getValue()
        
        fileNameAtt = flowFile.getAttribute("filename") + ".txt"

        image = Image.open(io.BytesIO(flowFile.getContentsAsBytes())).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        myAttributes = {"mime.type": "text/plain", "filename": fileNameAtt}
        
        if not outputAttributeName:
          myAttributes[outputAttributeName] = str.encode(generated_text)

        return FlowFileTransformResult(relationship = "success", contents = str.encode(generated_text), attributes=myAttributes)

    def getPropertyDescriptors(self):
        return self.descriptors