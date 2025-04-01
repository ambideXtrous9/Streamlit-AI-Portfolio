import torch
import streamlit as st 
from torchvision import transforms
from PIL import Image
from ImageClassifier import config
import time
from ImageClassifier.EfficientNetB0 import EfficientNet
from ImageClassifier.Xception import XceptionNet
from ImageClassifier.MobilenetV2 import MobileNetV2
from ImageClassifier.InceptionV3 import InceptionV3
from icons import glowingImgClassifier
import warnings
warnings.filterwarnings("ignore")


resize = transforms.Resize(size=(config.WIDTH,config.HEIGHT))

nrml = transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])  # Imagenet standards

transform_norm = transforms.Compose([
            transforms.ToTensor(),
            resize,
            nrml])

# Define the classes dictionary
class_labels = {
    'Adidas': 0, 'Apple': 1, 'BMW': 2, 'Citroen': 3, 'Cocacola': 4, 
    'DHL': 5, 'Fedex': 6, 'Ferrari': 7, 'Ford': 8, 'Google': 9, 
    'HP': 10, 'Heineken': 11, 'Intel': 12, 'McDonalds': 13, 'Mini': 14, 
    'Nbc': 15, 'Nike': 16, 'Pepsi': 17, 'Porsche': 18, 'Puma': 19, 
    'RedBull': 20, 'Sprite': 21, 'Starbucks': 22, 'Texaco': 23, 
    'Unicef': 24, 'Vodafone': 25, 'Yahoo': 26
}

# Reverse the dictionary to map indices to class names
index_to_class = {v: k for k, v in class_labels.items()}


def model_init():
    xcep_model = XceptionNet(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('ImageClassifier/Xception.ckpt',map_location=torch.device('cpu'))
    xcep_model.load_state_dict(checkpoint['state_dict'])
    
    mobile_model = MobileNetV2(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('ImageClassifier/MobileNetV2.ckpt',map_location=torch.device('cpu'))
    mobile_model.load_state_dict(checkpoint['state_dict'])
    
    incep_model = InceptionV3(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('ImageClassifier/InceptionV3.ckpt',map_location=torch.device('cpu'))
    incep_model.load_state_dict(checkpoint['state_dict'])
    
    effcent_model = EfficientNet(num_classes=config.NUM_CLASSES,lr=config.LR)
    checkpoint = torch.load('ImageClassifier/EfficientNet.ckpt',map_location=torch.device('cpu'))
    effcent_model.load_state_dict(checkpoint['state_dict'])
    
    return [("Xception",xcep_model),("InceptionV3",incep_model),("MobileNetV2", mobile_model),("EfficientNet" ,effcent_model)]
    

models = model_init()



# Function to calculate and print the model size and number of parameters
def model_details(model):
    # Calculate model size
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()

    size_all_mb = (param_size + buffer_size) / 1024**2  # Convert to megabytes

    # Calculate total number of parameters in millions
    total_params = sum(p.numel() for p in model.parameters())
    total_params_million = total_params / 10**6  # Convert to millions
    

    st.markdown(f"📚 Size: **{size_all_mb:.2f}** MB")
    st.markdown(f"💡 Parameters: **{total_params_million:.2f} M**")
    
    
    
def inference(cnnmodel,image_path):
        
    image = Image.open(image_path).convert('RGB')
    
    cnnmodel[1].eval()
    input_tensor = transform_norm(image).unsqueeze(0)  # Add batch dimension
    
    # Measure inference time
    start_time = time.time()
    
    with torch.no_grad():
        output = cnnmodel[1](input_tensor)
        probs = torch.exp(output)
        # Get the predicted class index
        index = torch.argmax(output).item()
        
    end_time = time.time()
    inference_time = end_time - start_time
        
    predicted_class = index_to_class[index]
    predprob = round(probs[0][index].item(),2)
    if(predprob < 0.80) : predicted_class = 'None'
    
    st.markdown(f"🚀 Model: **{cnnmodel[0]}**")
    model_details(cnnmodel[1])
    st.markdown(f"🌱 Predicted Class: **{predicted_class}**")
    st.markdown(f"🎃 Accuracy: **{predprob:.2f}**")
    st.markdown(f"🌟 Inference Time: **{inference_time:.4f} seconds**")
    


def model_card():
    
    st.write("""
    ### 🚀 **Exploring Image Classification Models with Transfer Learning!**
    - 🆕 **Models Evaluated**: We've assessed the following state-of-the-art models using the Flick27 dataset:
        - **Xception** 🤖
        - **InceptionV3** 🌈
        - **MobileNetV2** 📱
        - **EfficientNet** ⚡

    ### 🔍 **Performance Aspects Compared**
    - **⚡ Inference Time**: How quickly each model makes predictions.
    - **📦 Model Size**: The storage requirements of each model.
    - **🔢 Number of Parameters**: The complexity and capacity of each model.
    - **📈 Accuracy**: How well each model performs in classifying images.
    - **🏷️ Predicted Class**: The class each model predicts for the input images.

    - 📚 **Understanding Trade-Offs**: These comparisons will help us understand the trade-offs between speed, efficiency, and accuracy for each model.

    💡 **You can train your own model on any dataset following the link below:**
    [Train Your Model](https://github.com/ambideXtrous9/Brand-Logo-Classification-using-TransferLearning-Flickr27/tree/main/Final%20Model)
    
    🚀 **To see How it works..**
    """)


    
    imgpath = st.file_uploader("Upload an Image containing Brand Logo", type=["jpg", "jpeg", "png"])
    
    
    if imgpath is not None:
        
        st.image(imgpath, caption="Uploaded Image", use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)

        # Card 1: Model Name
        with col1:
            inference(models[0],image_path=imgpath)
        

        # Card 2: Model Details
        with col2:
            inference(models[1],image_path=imgpath)
            

        # Card 3: Predicted Class
        with col3:
            inference(models[2],image_path=imgpath)
            

        # Card 4: Inference Time
        with col4:
            inference(models[3],image_path=imgpath)
            
    glowingImgClassifier()
        

