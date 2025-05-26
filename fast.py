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

image_class_names= ['cat' , 'dog']



weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT

loaded_model = torchvision.models.efficientnet_b0(weights=weight)

loaded_model.classifier = nn.Linear(in_features=1280, out_features=len(image_class_names))

loaded_model.load_state_dict(torch.load('cat.pth',  map_location=torch.device('cpu')))
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
