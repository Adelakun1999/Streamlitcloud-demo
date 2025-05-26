import torchvision
import streamlit as st 
from PIL import Image 
from torchvision import transforms
import torch
import torchvision
from torch import nn 

class_names = ['cat' , 'dog']

st.title('Image classifcation')

weight = torchvision.models.EfficientNet_B0_Weights.DEFAULT

loaded_model = torchvision.models.efficientnet_b0(weights = weight)

transforms = weight.transforms()


loaded_model.classifier = nn.Linear(in_features=1280 ,  out_features = len(class_names))

loaded_model.load_state_dict(torch.load(f='cat.pth', weights_only=True,  map_location=torch.device('cpu')))
loaded_model.eval()



uploaded_file = st.file_uploader('choose an image' , type=['jpg','jpeg','png'])

if uploaded_file is not None :
    image =Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    image = transforms(image).unsqueeze(dim=0)

    st.write('The shape of the image is    :  ' ,  image.shape)

    with torch.inference_mode():
        logit = loaded_model(image)
        y_label = logit.argmax(dim=1)


    if st.button('Predict the class of the image'):
        with st.spinner('Generating Model prediction'):
            st.success(f'Predicted Class of the Image : {class_names[y_label]}')
            st.balloons()





    
