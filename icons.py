import streamlit as st 


def glowingLogo(href, iconlink, size=50, R=64, G=224, B=208):
    return f'''
        <div class="inner-container">
            <a href='{href}' target='_blank'>
                <img src='{iconlink}' class='glowing-image'/>
            </a>
        </div>
    '''

def glowingSocial():
    # Define the glowing logos
    kaggle = glowingLogo('https://www.kaggle.com/sushovansaha9', 'https://cdn4.iconfinder.com/data/icons/logos-and-brands/512/189_Kaggle_logo_logos-512.png', size=50, R=100)
    github = glowingLogo('https://github.com/ambideXtrous9', 'https://qph.cf2.quoracdn.net/main-qimg-729a22aba98d1235fdce4883accaf81e', size=50, G=100)
    linkedin = glowingLogo('https://www.linkedin.com/in/sushovan-saha-29a00a113', 'https://images.rawpixel.com/image_png_800/czNmcy1wcml2YXRlL3Jhd3BpeGVsX2ltYWdlcy93ZWJzaXRlX2NvbnRlbnQvbHIvdjk4Mi1kMy0xMC5wbmc.png', size=50, B=100)
    leetcode = glowingLogo('https://leetcode.com/u/ambideXtrous9/', 'https://upload.wikimedia.org/wikipedia/commons/8/8e/LeetCode_Logo_1.png', size=50)
    medium = glowingLogo('https://medium.com/@sushovansaha95', 'https://cdn4.iconfinder.com/data/icons/social-media-2210/24/Medium-512.png', size=50)

    # Combine all logos into a footer
    st.markdown(f'''
        <style>
        @keyframes glow {{
            0% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
            50% {{ box-shadow: 0 0 20px 10px rgba(64, 224, 208, 1); }}
            100% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
        }}

        .glowing-image {{
            width: 50px;
            height: 50px;
            border-radius: 50%;
            animation: glow 2s infinite;
        }}

        .footer {{
            display: flex;
            justify-content: center;
            gap: 20px;
            margin-top: 100px;
            padding: 0px;
            width: 100%;
            position: relative;
            bottom: 0;
        }}

        .inner-container {{
            display: inline-block;
        }}
        </style>

        <div class="footer">
            {kaggle}
            {github}
            {linkedin}
            {leetcode}
            {medium}
        </div>
    ''', unsafe_allow_html=True)
    
    
    
def glowingYolo():
    
    iconlink = 'https://miro.medium.com/v2/resize:fit:4800/format:webp/1*x4eteo0X9VqrLEHAYyCX8Q.jpeg'
    proflink = 'https://medium.com/@sushovansaha95/yolov8-1-on-custom-dataset-logo-detection-8915286999ef'
    
    Yolo = glowingLogo(proflink, iconlink, size=50, R=100)
   
    # Combine all logos into a footer
    st.markdown(f'''
        <style>
        @keyframes glow {{
            0% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
            50% {{ box-shadow: 0 0 20px 10px rgba(64, 224, 208, 1); }}
            100% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
        }}

        .glowing-image {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            animation: glow 2s infinite;
        }}

        .footer {{
            display: flex;
            justify-content: right;
            gap: 50px;
            margin-top: 50px;
            padding: 10px;
            width: 100%;
            position: relative;
            bottom: 0;
        }}

        .inner-container {{
            display: inline-block;
        }}
        </style>

        <div class="footer">
            {Yolo}
        </div>
    ''', unsafe_allow_html=True)


def glowingLLM():
    
    iconlink = 'https://pbs.twimg.com/media/GG3sJg7X0AEaNIq.jpg'
    proflink = 'https://medium.com/@sushovansaha95/finetuning-gemma2b-instruct-for-qa-with-rag-6d879226157b'
    
    gemma = glowingLogo(href=proflink,iconlink=iconlink)
    

    iconlink = 'https://aidrivensearch.com/wp-content/uploads/2023/02/T5-framework-Logo-280px.png'
    proflink = 'https://github.com/ambideXtrous9/MTP-News-Article-based-Question-Answering-System'
    
    t5 = glowingLogo(href=proflink,iconlink=iconlink)
    
   
    # Combine all logos into a footer
    st.markdown(f'''
        <style>
        @keyframes glow {{
            0% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
            50% {{ box-shadow: 0 0 20px 10px rgba(64, 224, 208, 1); }}
            100% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
        }}

        .glowing-image {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            animation: glow 2s infinite;
        }}

        .footer {{
            display: flex;
            justify-content: right;
            gap: 50px;
            margin-top: 50px;
            padding: 10px;
            width: 100%;
            position: relative;
            bottom: 0;
        }}

        .inner-container {{
            display: inline-block;
        }}
        </style>

        <div class="footer">
            {gemma}
            {t5}
        </div>
    ''', unsafe_allow_html=True)




def glowingCluster():
    
    iconlink = 'https://media.licdn.com/dms/image/C4E12AQGopXR2OLgA9Q/article-cover_image-shrink_600_2000/0/1537440291948?e=2147483647&v=beta&t=ciX9-639Lwc27E74CNPBKN2gx9hwRr5-b1b8y2sLNOQ'
    proflink = 'https://github.com/ambideXtrous9/Learning-Python/blob/main/ML%20and%20NN/Clustering.ipynb'
    
    Yolo = glowingLogo(proflink, iconlink, size=50, R=100)
   
    # Combine all logos into a footer
    st.markdown(f'''
        <style>
        @keyframes glow {{
            0% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
            50% {{ box-shadow: 0 0 20px 10px rgba(64, 224, 208, 1); }}
            100% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
        }}

        .glowing-image {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            animation: glow 2s infinite;
        }}

        .footer {{
            display: flex;
            justify-content: right;
            gap: 50px;
            margin-top: 50px;
            padding: 10px;
            width: 100%;
            position: relative;
            bottom: 0;
        }}

        .inner-container {{
            display: inline-block;
        }}
        </style>

        <div class="footer">
            {Yolo}
        </div>
    ''', unsafe_allow_html=True)
    
    
def glowingImgClassifier():
    
    iconlink = 'https://img.freepik.com/premium-photo/artificial-intelligence-neural-networks-background_1106493-34240.jpg'
    proflink = 'https://github.com/ambideXtrous9/Brand-Logo-Classification-using-TransferLearning-Flickr27'
    
    Yolo = glowingLogo(proflink, iconlink, size=50, R=100)
   
    # Combine all logos into a footer
    st.markdown(f'''
        <style>
        @keyframes glow {{
            0% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
            50% {{ box-shadow: 0 0 20px 10px rgba(64, 224, 208, 1); }}
            100% {{ box-shadow: 0 0 10px 5px rgba(64, 224, 208, 0.8); }}
        }}

        .glowing-image {{
            width: 80px;
            height: 80px;
            border-radius: 50%;
            animation: glow 2s infinite;
        }}

        .footer {{
            display: flex;
            justify-content: right;
            gap: 50px;
            margin-top: 50px;
            padding: 10px;
            width: 100%;
            position: relative;
            bottom: 0;
        }}

        .inner-container {{
            display: inline-block;
        }}
        </style>

        <div class="footer">
            {Yolo}
        </div>
    ''', unsafe_allow_html=True)
    
    