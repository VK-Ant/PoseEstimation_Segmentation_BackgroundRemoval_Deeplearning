import gradio as gr
from transformers import SegformerFeatureExtractor, SegformerForSemanticSegmentation
from PIL import Image
import requests
import numpy as np
import random
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
from PIL import Image
import requests

feature_extractor = SegformerImageProcessor.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")
model = SegformerForSemanticSegmentation.from_pretrained("nvidia/segformer-b0-finetuned-ade-512-512")

# Define a function to apply random colors to the segmentation map
def apply_random_colors(segmentation_map):
    num_classes = model.config.num_labels
    colors = [[random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)] for _ in range(num_classes)]
    colored_map = np.zeros(segmentation_map.shape + (3,), dtype=np.uint8)

    for label in range(num_classes):
        colored_map[segmentation_map == label] = colors[label]

    return colored_map

def semantic_segmentation(input_image):
    image = input_image.copy()  # Make a copy to ensure PIL image is not modified

    inputs = feature_extractor(images=image, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    predictions = np.argmax(logits[0].detach().numpy(), axis=0)
    
    # Resize the segmented map to 512x512
    resized_predictions = Image.fromarray(predictions.astype(np.uint8)).resize((512, 512))
    colored_map = apply_random_colors(np.array(resized_predictions))

    return colored_map

# Define the Gradio interface
inputs = gr.inputs.Image(type="pil", label="Input Image")
outputs = gr.outputs.Image(type="numpy", label="Segmentation Map")

iface = gr.Interface(fn=semantic_segmentation, inputs=inputs, outputs=outputs, title="VK Semantic Segmentation", 
                     description="Cityscape semantic segmentation (512x512)")

iface.launch()
