from pydantic import BaseModel
from PIL import Image
import os 
from fastapi.responses import JSONResponse
from fastapi import FastAPI , UploadFile, File
import io
from torchvision import transforms
import torch
import torchvision
import requests
from torch import nn
import uvicorn
import sys


image_class_names = ['NOT SAFE', 'SAFE', 'QUESTIONABLE']


weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT

loaded_model = torchvision.models.efficientnet_b0(weights=weight)

loaded_model.classifier = nn.Linear(in_features=1280, out_features=len(image_class_names))

loaded_model.load_state_dict(torch.load('vision.pth',  map_location=torch.device('cpu')))
loaded_model.eval()
transforms = weight.transforms()

app = FastAPI(
     title='SingleEye Image classification',
     description=(
        "This API allows users to upload an image (in ,jpg, JPEG or PNG format) "
        "for classification into one of three categories: SAFE, NOT SAFE, or QUESTIONABLE."),
         version="1.0.0")

@app.get('/', 
         tags=["General"],
         summary="API Health Check",
         description="Returns a simple message to confirm the API is running.")


def get():
    return {'Message ' : 'Single eye Image classification'}



# Pydantic model for Url input  


@app.post("/predict-image", tags=["Image Classification"])
async def predict_image(file: UploadFile = File(...)):
    try:
        if file.content_type not in ["image/jpeg", "image/png"]:
            return JSONResponse(content={"error": "Only JPEG and PNG images are allowed."}, status_code=400)

        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")


        input_tensor = transforms(image).unsqueeze(0)

        with torch.no_grad():
            logits = loaded_model(input_tensor)
            probabilities = torch.softmax(logits, dim=1)

        confidence, predicted_class = probabilities.max(dim=1)

        return {
            "Predicted Class": image_class_names[predicted_class.item()],
            "Confidence": round(confidence.item(), 3)
        }

    except Exception as e:
        return JSONResponse(content={"error": f"An error occurred: {str(e)}"}, status_code=500)


class ImageURL(BaseModel):
    image_url : str
    model_config = {
        "json_schema_extra" : {
            "example" : {
                "image_url" : "https://example.com/image.jpg" 
            }
        }
    }



#Endpoint for URL-based Image Prediction

@app.post("/predict-image-url",
           tags=["Image Classification/Text Classification"],
        summary="Classify an Image",
        description=(
        "Upload the image url to classify it as "
        "'SAFE', 'NOT SAFE', or 'QUESTIONABLE'. The image will be preprocessed and "
        "evaluated by the Deep learning model, which outputs the predicted label."
    ),
    response_description="The predicted label for the uploaded image." )


async def predict_image_url(image_url: ImageURL):
    try:
        # Fetch the image from the provided URL
        response = requests.get(image_url.image_url)
        response.raise_for_status()  # Ensure the request was successful
        
        # Load the image into PIL
        image = Image.open(io.BytesIO(response.content))
        image = image.convert('RGB')
        
        # Preprocess the image and add a batch dimension
        input_tensor = transforms(image).unsqueeze(dim=0)

        # Run inference
        with torch.inference_mode():
            y_logit = loaded_model(input_tensor)
            probabilities = y_logit.softmax(dim=1)

        confidence , predicted_class = probabilities.max(dim=1)
        
        
        
        
        return {"image_url" : image_url.image_url,
                'Predicted_Class_Image' : image_class_names[predicted_class.item()],
                'Confidence_Image' : round(confidence.item() ,3)}
    
    except Exception as e:
        return JSONResponse(
            content={"error": f"An error occurred: {str(e)}"},
            status_code=500
        )
    