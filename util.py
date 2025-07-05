import streamlit as st 
import requests
from st_social_media_links import SocialMediaIcons
from LogoYolo.inference import predict
from PIL import Image
from NewsQALLM.chatbot import ChatBot
from icons import glowingSocial,glowingYolo,glowingLLM


def Social(sidebarPos = False,heading = None):
    
    if heading != None:
        st.title(f":rainbow[{heading}]")
        
    social_media_links = [
            "https://www.linkedin.com/in/sushovan-saha-29a00a113",
            "https://github.com/ambideXtrous9",
            "https://medium.com/@sushovansaha95"]

    social_media_icons = SocialMediaIcons(social_media_links) 

    social_media_icons.render(sidebar=sidebarPos, justify_content="center")

def HomePage():
    
    st.title(":blue[My Portfolio] 🤓")
    
    
    gif_path = 'thor.gif'
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.image(gif_path)
    
    # Display "About Me" text in the right column
    with col2:
        st.subheader("🌱 About Me")
        st.write("""
        👋 Hi there! I'm **Sushovan Saha**, a Machine Learning (ML) enthusiast specializing in **Natural Language Processing (NLP)** and **Computer Vision (CV)**. 
        I did my M.Tech in Data Science from **IIT Guwahati**. I am also a **Kaggle Notebook Expert**.
    
        🌟 I'm passionate about exploring the possibilities of ML to solve real-world problems and improve people's lives. 
        I love working on challenging projects that require me to stretch my abilities and learn new things.
    
        📚 In my free time, I like to contribute in **Kaggle**, Write ML blogs in **Medium**, read ML related blogs and updates. 
        I'm always looking for ways to stay up-to-date with the latest developments in the field.
        """)
        
    glowingSocial()
    
    
    


def GitHubStats():
    st.title(":rainbow[GitHub Stats]")
    username = "ambideXtrous9"  # Replace with your GitHub username
    response = requests.get(f"https://api.github.com/users/{username}")

    if response.status_code == 200:
        user_data = response.json()
        st.write(f"**Username:** {user_data['login']}")
        st.write(f"**Name:** {user_data.get('name', 'N/A')}")
        st.write(f"**Public Repos:** {user_data['public_repos']}")
        st.write(f"**Followers:** {user_data['followers']}")
        st.write(f"**Following:** {user_data['following']}")
        st.write(f"**Profile URL:** {user_data['html_url']}")
    else:
        st.error("Failed to fetch GitHub stats. Please check the username or try again later.")

        
def YoloforLogo():
    
    st.write("""
            ### 🚀 **YOLOv8.1: The Latest in Object Detection**
            - 🆕 **YOLOv8.1 is out!**: The newest update in the YOLO series, maintaining its position as the state-of-the-art model for:
            - 🎯 **Object Detection**
            - 🌀 **Instance Segmentation**
            - 🏷️ **Classification**

            ### ⚠️ **Main Challenge: Custom Dataset Preparation**
            - 🔍 **Dataset Selection**: Using **Flickr27** as our image dataset.
            - 📸 **Flickr27 Overview**: Contains 27 different brand logos, perfect for training YOLO on custom data.

            - 💼 **Custom Dataset Prep**: The most crucial step in training YOLO models.

            - 🛠️ **Get Ready to Train**: With YOLOv8.1 and Flickr27, you'll be well-equipped to handle custom object detection tasks!
            """)
                

    uploaded_file = st.file_uploader("Upload an Image containing Brand Logo", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        
        uploaded_image = Image.open(uploaded_file)

        # Perform prediction
        prediction_image = predict(uploaded_image)
        
        # Create two columns for side-by-side images
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(uploaded_file, caption="Uploaded Image", width = 200,use_container_width='auto')
            
            
        with col2:
            st.image(prediction_image, caption="Predicted Image", width=200,use_container_width='auto')
            
    glowingYolo()
            
    
            
            
def NewsQA():
    
    ChatBot()
    #glowingLLM()
    