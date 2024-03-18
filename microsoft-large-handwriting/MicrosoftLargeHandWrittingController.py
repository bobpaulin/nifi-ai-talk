from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
import io
from nifiapi.properties import PropertyDescriptor
from nifiapi.properties import StandardValidators
from nifiapi.flowfiletransform import FlowFileTransform, FlowFileTransformResult

class MicrosoftLargeHandWrittingController(FlowFileTransform):
    class Java:
        implements = ['org.apache.nifi.python.processor.FlowFileTransform']
    class ProcessorDetails:
        version = '0.0.1-SNAPSHOT'


    def __init__(self, **kwargs):
        # Build Property Descriptors
        self.indentation = PropertyDescriptor(
            name="Indentation",
            description="Number of spaces",
            required = True,
            default_value="4",
            validators = [StandardValidators.NON_NEGATIVE_INTEGER_VALIDATOR]
        )
        self.descriptors = [self.indentation]
        self.processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
        self.model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
        

    def transform(self, context, flowFile):
        spaces = context.getProperty(self.indentation.name).asInteger()
        
        image = Image.open(io.BytesIO(flowFile.getContentsAsBytes())).convert("RGB")
        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values
        generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        return FlowFileTransformResult(relationship = "success", contents = str.encode(generated_text))


    def getPropertyDescriptors(self):
        return self.descriptors